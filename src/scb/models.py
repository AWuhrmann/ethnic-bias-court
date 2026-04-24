from pydantic import BaseModel, computed_field


class NameGroup(BaseModel):
    origin: str
    male_first_names: list[str]
    female_first_names: list[str]
    last_names: list[str]

    @computed_field
    @property
    def first_names(self) -> list[str]:
        return self.male_first_names + self.female_first_names


class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class Substitution(BaseModel):
    placeholder: str
    first_name: str
    last_name: str


class DeanonymizedDoc(BaseModel):
    original: Document
    deanonymized_text: str
    substitutions: list[Substitution]
    name_origin: str
