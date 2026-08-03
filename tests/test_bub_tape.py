import asyncio

from bub.builtin.store import FileTapeStore
from bub.builtin.tape import Tape
from bub.tape import AsyncTapeStoreAdapter, TapeContext, TapeEntry


def test_handoff_preserves_history_and_changes_view_origin(tmp_path) -> None:
    async def exercise() -> None:
        store = AsyncTapeStoreAdapter(FileTapeStore(tmp_path))
        tape = Tape(tmp_path / "archive", store, TapeContext()).scoped("demo__test")
        await tape.ensure_bootstrap_anchor()
        await store.append(tape.name, TapeEntry.message({"role": "user", "content": "before"}))
        await tape.handoff(name="phase/2", state={"summary": "checkpoint"})
        await store.append(tape.name, TapeEntry.message({"role": "user", "content": "after"}))

        all_entries = list(await store.fetch_all(tape.query()))
        assert any(entry.payload.get("content") == "before" for entry in all_entries)
        assert any(entry.kind == "anchor" and entry.payload.get("name") == "phase/2" for entry in all_entries)

        visible = await tape.read_messages()
        assert visible == [{"role": "user", "content": "after"}]

    asyncio.run(exercise())
