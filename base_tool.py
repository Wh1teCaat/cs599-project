from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.tools import BaseTool


class BaseToolWrapper(ABC):
    """所有自定义工具的抽象基类"""

    DEFAULT_NAME: Optional[str] = None
    DEFAULT_DESC: Optional[str] = None

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        self.name = name or self.DEFAULT_NAME
        self.description = description or self.DEFAULT_DESC

    @abstractmethod
    def build(self) -> BaseTool:
        pass
