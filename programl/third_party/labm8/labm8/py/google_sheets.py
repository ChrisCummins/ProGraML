"""Module for working with google sheets."""
import pathlib
from typing import Optional

import appdirs
import gspread
import gspread_dataframe
import pandas as pd
from labm8.py import app
from oauth2client import service_account

app.DEFINE_output_path(
    "google_sheets_credentials",
    pathlib.Path(appdirs.user_config_dir("phd", appauthor="Chris Cummins"))
    / "google_sheets_credentials.json",
    "The path to a google service account credentials JSON file.",
)
app.DEFINE_string(
    "google_sheets_default_share_with",
    "chrisc.101@gmail.com",
    "The default email adress to share google sheets with.",
)

FLAGS = app.FLAGS


class GoogleSheets:
    """An object for working with google sheets."""

    def __init__(self, credentials_file: pathlib.Path):
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]

        credentials = service_account.ServiceAccountCredentials.from_json_keyfile_name(
            str(credentials_file), scope
        )

        self._connection = gspread.authorize(credentials)

    def GetSpreadsheet(self, name: str):
        """Return the spreadsheet with the given name.

        Args:
          name: The name of the spreadsheet, as it appears in Google Drive.

        Returns:
          A spreadsheet object.

        Raises:
          gspread.exceptions.SpreadsheetNotFound: If the given spreadsheet is not
            found.
        """
        return self._connection.open(name)

    def GetOrCreateSpreadsheet(
        self, name: str, share_with_email_address: Optional[str] = None
    ):
        """Return the speadsheet with the given name, creating it if necessary
        and sharing it with the given email address."""
        share_with_email_address = (
            share_with_email_address or FLAGS.google_sheets_default_share_with
        )
        try:
            sheet = self.GetSpreadsheet(name)
        except gspread.exceptions.SpreadsheetNotFound:
            sheet = self._connection.create(name)
            sheet.share(share_with_email_address, perm_type="user", role="writer")
        return sheet

    @staticmethod
    def GetWorksheet(sheet, name: str):
        """Return the worksheet with the given name, creating it if necessary."""
        return sheet.worksheet(name)

    @classmethod
    def GetOrCreateWorksheet(cls, sheet, name: str):
        """Return the worksheet with the given name, creating it if necessary."""
        try:
            return cls.GetWorksheet(sheet, name)
        except gspread.exceptions.WorksheetNotFound:
            return sheet.add_worksheet(title=name, rows=1, cols=1)

    @staticmethod
    def ExportDataFrame(worksheet, df: pd.DataFrame, index: bool = True) -> None:
        """Export the given dataframe to a worksheet."""

        gspread_dataframe.set_with_dataframe(
            worksheet, df, include_index=index, resize=True
        )

    @staticmethod
    def GetAsDataFrame(worksheet, **opts) -> pd.DataFrame:
        """Return a worksheet as a dataframe.

        Args:
          worksheet: The name of the worksheet to return.
          **opts: Keyword arguments passed to gspread_dataframe.get_as_data_frame().
            See: https://pythonhosted.org/gspread-dataframe/#id1

        Returns:
          A pandas dataframe.
        """
        return gspread_dataframe.get_as_dataframe(worksheet, **opts)

    @classmethod
    def FromFlagsOrDie(cls) -> "GoogleSheets":
        try:
            return cls(FLAGS.google_sheets_credentials)
        except Exception as e:
            app.FatalWithoutStackTrace("%s", e)
