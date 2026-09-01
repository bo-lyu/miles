"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.utils.http_utils import post, post_bytes_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
        agent_router_url: str | None = None,
    ):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        # What the agent function hands its agent, which may run outside the cluster.
        # The driver's own calls stay on base_url and off any external path.
        self.agent_base_url = f"{agent_router_url or router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        # The samples-wire allowlist must match the server's encode: v1 default,
        # extended under --use-session-server v2 (create() selects from args;
        # direct constructions keep v1).
        self.samples_wire_fields = samples_wire_fields

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance owning this session, as recorded in sample metadata."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(args: Namespace):
        session_addrs = getattr(args, "session_server_addrs", None)
        if not session_addrs:
            raise RuntimeError(
                "session_server_addrs is not set. Pass --use-session-server to start the session server."
            )
        # Positionally aligned with session_addrs; absent before the driver
        # records it, and then the cluster address serves the agent too.
        external_addrs = getattr(args, "session_server_external_addrs", None) or session_addrs
        assert len(external_addrs) == len(session_addrs), f"{external_addrs=} misaligned with {session_addrs=}"
        # The only routing decision in the system: pick the owning instance once
        # per session; every later touch of the session reuses this URL. Both URLs
        # name that one instance, so the agent and the driver never split across two.
        instance_index = random.randrange(len(session_addrs))
        session_addr = session_addrs[instance_index]
        session_url = f"http://{session_addr}"
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        session_server_instance_id = instance_ids.get(session_addr)
        response = await post(f"{session_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
            samples_wire_fields=COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS,
            agent_router_url=f"http://{external_addrs[instance_index]}",
        )

    async def collect_samples(
        self, input_sample: Sample, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> SamplesReply:
        """Fetch server-assembled training samples for this session."""
        body: dict = {"max_seq_len": max_seq_len}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        try:
            # Timeouts and transport errors propagate after cleanup, for `generate` to handle.
            payload = await post_bytes_no_retry(
                f"{self.base_url}/samples",
                body,
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            try:
                await asyncio.wait_for(
                    post(self.base_url, {}, action="delete"),
                    timeout=_SESSION_REQUEST_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Failed to delete session {self.session_id} after collecting samples: {e}")

        return decode_samples_and_merge_input_sample(payload, input_sample, fields=self.samples_wire_fields)
