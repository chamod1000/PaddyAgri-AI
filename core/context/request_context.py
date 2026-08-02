"""
RequestContext Module - Version 3.0 Architecture
Captures HTTP/UI session metadata, user location, and upload state.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class RequestContext(BaseModel):
    """Immutable context capturing request-level metadata."""
    session_id: str = Field(..., description="Unique session identifier")
    request_id: str = Field(..., description="Unique request correlation ID")
    trace_id: str = Field(..., description="Distributed trace identifier")
    district: Optional[str] = Field(default=None, description="User selected district/location")
    has_image: bool = Field(default=False, description="True if image binary attached to request")
    image_bytes: Optional[bytes] = Field(default=None, description="Raw image bytes if attached")
    image_hash: Optional[str] = Field(default=None, description="SHA256 hash of image for cache keying")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)
