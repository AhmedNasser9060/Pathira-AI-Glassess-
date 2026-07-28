from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field


class ProfileUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(default=None, max_length=20)
    avatar: Optional[str] = Field(
        default=None,
        max_length=350_000,
        description="Base64-encoded image, ~250 KB raw max. Will move to S3 URL in a future phase.",
    )

    def has_any(self) -> bool:
        return any(v is not None for v in (self.name, self.email, self.phone, self.avatar))


class SetTypeRequest(BaseModel):
    user_type: Literal["main", "companion"]


class SettingsUpdate(BaseModel):
    notifications: Optional[bool] = None
    bluetooth: Optional[bool] = None
    live_vision: Optional[bool] = None
    location_tracking: Optional[bool] = None
    voice_commands: Optional[bool] = None

    def to_partial_dict(self) -> dict:
        # Only include keys the caller actually sent.
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SettingsResponse(BaseModel):
    settings: dict


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1)


class DeviceTokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=512)
    platform: Literal["ios", "android"]
