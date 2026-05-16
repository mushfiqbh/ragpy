import os
import io
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build # type: ignore
from googleapiclient.http import MediaIoBaseDownload # type: ignore
from google.oauth2 import service_account # type: ignore
from dotenv import load_dotenv

from pipeline.parser import parse_document
from pipeline.chunker import chunk_text
from pipeline.embedder import embed
from pipeline.store import default_store

load_dotenv()

class DriveService:
    def __init__(self):
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path and os.path.exists(credentials_path):
            self.creds = service_account.Credentials.from_service_account_file(
                credentials_path, 
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            self.service = build("drive", "v3", credentials=self.creds)
        else:
            raise Exception("Google credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable.")

    def list_files(self, folder_id: str) -> List[Dict[str, Any]]:
        """List all files in a folder recursively or non-recursively"""
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType, md5Checksum, modifiedTime)"
        ).execute()
        return results.get("files", [])

    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_course_code(self, text: str) -> Optional[str]:
        """Extract course code like CSE-111, MAT 101, etc. from text"""
        if not text:
            return None
        # Match pattern like XXX-123 or XXX 123
        pattern = r"([A-Z]{2,4})\s*[-]?\s*(\d{3,4})"
        match = re.search(pattern, text.upper())
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return None

    def download_file(self, file_id: str, filename: str, mime_type: str, output_dir: Path) -> Optional[Path]:
        """Download or export a file from Google Drive"""
        filepath = output_dir / filename
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Handle Google workspace types (Docs, Sheets, etc.)
            is_google_type = mime_type.startswith("application/vnd.google-apps")
            is_exportable = is_google_type and mime_type in [
                "application/vnd.google-apps.document",
                "application/vnd.google-apps.spreadsheet",
                "application/vnd.google-apps.presentation",
                "application/vnd.google-apps.drawing"
            ]

            if is_exportable:
                # Map to PDF for most types, CSV for spreadsheets
                export_mime = "application/pdf"
                if mime_type == "application/vnd.google-apps.spreadsheet":
                    export_mime = "text/csv"
                    if not filename.endswith(".csv"):
                        filepath = filepath.with_suffix(".csv")
                elif not filename.endswith(".pdf"):
                    filepath = filepath.with_suffix(".pdf")
                
                request = self.service.files().export_media(fileId=file_id, mimeType=export_mime)
            elif is_google_type:
                # Other google types like folders or shortcuts
                return None
            else:
                # Regular binary files
                request = self.service.files().get_media(fileId=file_id)

            fh = io.FileIO(str(filepath), "wb")
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.close()
            
            return filepath
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None

    def process_folder(self, folder_id: str):
        """Recursively download and index files from a folder"""
        # Try to extract course code from folder name if course_code is GENERAL
        folder_metadata = self.service.files().get(fileId=folder_id, fields="name").execute()
        folder_name = folder_metadata.get("name", "")
        
        extracted_code = self.extract_course_code(folder_name)
        current_course_code = extracted_code if extracted_code else "GENERAL"

        items = self.list_files(folder_id)
        temp_dir = Path("downloads")
        temp_dir.mkdir(exist_ok=True)
        
        results = []

        for item in items:
            file_id = item["id"]
            name = item["name"]
            mime = item["mimeType"]

            if mime == "application/vnd.google-apps.folder":
                # Recurse with current course code
                sub_results = self.process_folder(file_id)
                results.extend(sub_results)
            else:
                # Try to extract course code from file name as well
                file_course_code = self.extract_course_code(name)
                final_course_code = file_course_code if file_course_code else current_course_code

                # Deduplication check for regular files (Google Drive already provides MD5)
                md5_checksum = item.get("md5Checksum")
                if md5_checksum:
                    existing = default_store.supabase.table("documents").select("id").eq("checksum", md5_checksum).execute()
                    if existing.data:
                        print(f"Skipping {name} (ID: {file_id}) - checksum {md5_checksum} already exists.")
                        results.append({"filename": name, "status": "skipped", "reason": "duplicate"})
                        continue

                # Download
                print(f"Processing {name} ({file_id})")
                path = self.download_file(file_id, name, mime, temp_dir)
                
                if path and path.exists():
                    try:
                        # For Google Workspace files, we might need to calculate checksum after download (as they don't have md5Checksum in Drive API)
                        final_checksum = md5_checksum
                        if not final_checksum:
                            final_checksum = self.calculate_checksum(path)
                            # Check again after calculation for exportable files
                            existing = default_store.supabase.table("documents").select("id").eq("checksum", final_checksum).execute()
                            if existing.data:
                                print(f"Skipping {name} (ID: {file_id}) - local checksum {final_checksum} already exists.")
                                results.append({"filename": name, "status": "skipped", "reason": "duplicate"})
                                continue

                        # Parse, Chunk, Embed, Store (Mirroring Node logic)
                        text = parse_document(str(path))
                        if text:
                            chunks = chunk_text(text)
                            if chunks:
                                added = default_store.add_chunks(
                                    chunks,
                                    embed_fn=embed,
                                    metadata={
                                        "filename": name,
                                        "mime_type": mime,
                                        "drive_file_id": file_id,
                                        "checksum": final_checksum,
                                        "modified_time": item.get("modifiedTime", ""),
                                        "course_code": final_course_code
                                    }
                                )
                                results.append({"filename": name, "chunks": added, "status": "success"})
                    except Exception as e:
                        print(f"Failed to process {name}: {e}")
                        results.append({"filename": name, "error": str(e), "status": "failed"})
                    finally:
                        # Cleanup
                        if path.exists():
                            path.unlink()
                else:
                    results.append({"filename": name, "status": "skipped"})
        
        return results

# Singleton instance
default_drive_service = DriveService()
