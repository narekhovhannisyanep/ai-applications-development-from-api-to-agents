from pydantic import BaseModel


class Address(BaseModel):
    country: str
    city: str
    street: str
    flat_house: str


class CreditCard(BaseModel):
    num: str
    cvv: str
    exp_date: str


class UserCreate(BaseModel):
    name: str
    surname: str
    email: str
    phone: str | None = None
    date_of_birth: str | None = None
    address: Address | None = None
    gender: str | None = None
    company: str | None = None
    salary: float | None = None
    about_me: str
    credit_card: CreditCard | None = None


class UserUpdate(BaseModel):
    name: str | None = None
    surname: str | None = None
    email: str | None = None
    phone: str | None = None
    date_of_birth: str | None = None
    address: Address | None = None
    gender: str | None = None
    company: str | None = None
    salary: float | None = None
    credit_card: UserCreate | None = None


class UserSearchRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    surname: str | None = None
    gender: str | None = None
