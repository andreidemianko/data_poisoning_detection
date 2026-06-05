from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import ZipConfig
from .utils import is_relative_to


class ArchiveError(RuntimeError):
    """
    Raised when an archive cannot be safely processed.
    """


class UnsafeArchiveError(ArchiveError):
    """
    Raised when an archive violates safety limits.
    """


@dataclass
class ResolvedInput:
    """
    Resolved scan input.

    path:
        Path to scan. It can be the original input path or a temporary
        extraction directory.

    cleanup_path:
        Temporary directory to delete after scanning. None means no cleanup is
        required.
    """

    path: Path
    cleanup_path: Path | None = None

    def cleanup(self) -> None:
        """
        Remove temporary files created during archive resolution.
        """

        if self.cleanup_path is None:
            return

        if self.cleanup_path.exists():
            shutil.rmtree(self.cleanup_path, ignore_errors=True)


class ArchiveResolver:
    """
    Resolve input paths before dataset discovery.

    Regular files and directories are returned unchanged. ZIP archives are
    safely extracted into a temporary directory and that directory is returned.
    """

    def __init__(self, config: ZipConfig | None = None) -> None:
        self.config = config or ZipConfig()

    def resolve(self, input_path: Path | str) -> ResolvedInput:
        """
        Resolve a file, directory, or supported archive into a scan path.
        """

        path = Path(input_path)

        if not path.exists():
            raise ArchiveError(f"Input path does not exist: {path}")

        if path.is_dir():
            return ResolvedInput(path=path)

        if path.is_file() and path.suffix.lower() == ".zip":
            return self._extract_zip(path)

        return ResolvedInput(path=path)

    def _extract_zip(self, path: Path) -> ResolvedInput:
        """
        Safely extract a ZIP archive into a temporary directory.
        """

        temp_root = Path(tempfile.mkdtemp(prefix="dataset_guard_zip_"))

        try:
            with zipfile.ZipFile(path, "r") as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]

                self._validate_member_count(members)
                self._validate_total_size(members)

                for member in members:
                    self._validate_member(path=path, member=member, destination_root=temp_root)
                    self._extract_member(archive=archive, member=member, destination_root=temp_root)

            return ResolvedInput(path=temp_root, cleanup_path=temp_root)

        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise

    def _validate_member_count(self, members: list[zipfile.ZipInfo]) -> None:
        """
        Validate the total number of files in the archive.
        """

        if len(members) > self.config.max_files:
            raise UnsafeArchiveError(
                f"ZIP archive contains too many files: {len(members)} > {self.config.max_files}"
            )

    def _validate_total_size(self, members: list[zipfile.ZipInfo]) -> None:
        """
        Validate the total uncompressed size of archive members.
        """

        total_uncompressed = sum(member.file_size for member in members)

        if total_uncompressed > self.config.max_total_uncompressed_bytes:
            raise UnsafeArchiveError(
                "ZIP archive uncompressed size is too large: "
                f"{total_uncompressed} > {self.config.max_total_uncompressed_bytes}"
            )

    def _validate_member(self, *, path: Path, member: zipfile.ZipInfo, destination_root: Path) -> None:
        """
        Validate one archive member before extraction.
        """

        member_name = member.filename

        if not member_name or member_name.endswith("/"):
            return

        member_path = Path(member_name)

        if member_path.is_absolute():
            raise UnsafeArchiveError(f"ZIP archive contains absolute path: {member_name}")

        normalized_destination = (destination_root / member_path).resolve()

        if not is_relative_to(normalized_destination, destination_root):
            raise UnsafeArchiveError(f"ZIP archive contains path traversal entry: {member_name}")

        if member.file_size > self.config.max_file_bytes:
            raise UnsafeArchiveError(
                f"ZIP member is too large: {member_name}: {member.file_size} > {self.config.max_file_bytes}"
            )

        compression_ratio = self._compression_ratio(member)

        if compression_ratio > self.config.max_compression_ratio:
            raise UnsafeArchiveError(
                f"ZIP member compression ratio is too high: "
                f"{member_name}: {compression_ratio:.2f} > {self.config.max_compression_ratio}"
            )

        # Keep this argument used for easier debugging/logging extension later.
        _ = path

    def _extract_member(
        self,
        *,
        archive: zipfile.ZipFile,
        member: zipfile.ZipInfo,
        destination_root: Path,
    ) -> None:
        """
        Extract a single validated member.

        This method avoids ZipFile.extract() so that path handling remains under
        our control.
        """

        destination_path = (destination_root / member.filename).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with archive.open(member, "r") as source:
            with destination_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    @staticmethod
    def _compression_ratio(member: zipfile.ZipInfo) -> float:
        """
        Return uncompressed_size / compressed_size.

        Empty compressed size can happen for empty files. Treat it safely.
        """

        if member.compress_size <= 0:
            if member.file_size <= 0:
                return 1.0

            return float("inf")

        return member.file_size / member.compress_size