"""Shared plumbing for the fusion page controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from pages.intravascular.popup_windows.message_boxes import ErrorMessage

if TYPE_CHECKING:
    from domain.fusion.runtime_types import FusionRuntimeData
    from pages.fusion.page import FusionPage


class StageController:
    """One stage of the fusion page. Reads state through the page (Clear All Data replaces
    page.data) and reaches other stages through their page attributes."""

    def __init__(self, page: FusionPage) -> None:
        self.page = page

    @property
    def data(self) -> FusionRuntimeData:
        return self.page.data

    @property
    def left_half(self):
        return self.page.left_half

    @property
    def right_half(self):
        return self.page.right_half

    @property
    def status_bar(self):
        return self.page.status_bar

    def require(self, ok: bool, message: str) -> bool:
        if not ok:
            ErrorMessage(self.page, message)
        return ok

    def run(self, busy_message: str, done_message: str, fn, *args, **kwargs):
        """Run a pipeline call with a status-bar message and a shared error path."""
        self.status_bar.showMessage(busy_message)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            logger.exception(f'Fusion pipeline step failed: {fn}')
            ErrorMessage(self.page, str(e))
            self.status_bar.showMessage('Failed — see log')
            return None
        self.status_bar.showMessage(done_message)
        return result
