import toml
from pydantic import BaseModel


class Project(BaseModel):
    name: str
    version: str

    @classmethod
    def from_toml(cls):
        with open("pyproject.toml") as f:
            config = toml.load(f)
        return cls(
            name=config["tool"]["poetry"]["name"],
            version=config["tool"]["poetry"]["version"],
        )

    @property
    def user_agent(self):
        return f"{self.name}/{self.version}"
