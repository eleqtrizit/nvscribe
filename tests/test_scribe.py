"""Tests for the scribe module."""

import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scribe import (
    UNKNOWN_SPEAKER,
    DiarizationSegment,
    MergedSegment,
    ScribeError,
    TranscriptionSegment,
    _create_diarizer,
    _create_transcriber,
    _find_ffmpeg,
    _prepare_audio,
    format_json,
    format_text,
    format_transcript_json,
    format_transcript_text,
    main,
    merge,
    run_diarization,
    run_transcription,
)


class TestMerge:
    """Tests for merge()."""

    def test_perfect_overlap(self):
        diarization = [DiarizationSegment("A", 0.0, 5.0)]
        transcription = [TranscriptionSegment("hello", 0.0, 5.0)]
        result = merge(diarization, transcription)
        assert result == [MergedSegment("A", "hello", 0.0, 5.0)]

    def test_partial_overlap_picks_best(self):
        diarization = [
            DiarizationSegment("A", 0.0, 3.0),
            DiarizationSegment("B", 2.0, 6.0),
        ]
        transcription = [TranscriptionSegment("hello", 1.0, 5.0)]
        result = merge(diarization, transcription)
        # B overlaps 2.0-5.0 = 3.0s, A overlaps 1.0-3.0 = 2.0s
        assert result[0].speaker == "B"

    def test_no_overlap_gives_unknown(self):
        diarization = [DiarizationSegment("A", 10.0, 15.0)]
        transcription = [TranscriptionSegment("hello", 0.0, 5.0)]
        result = merge(diarization, transcription)
        assert result[0].speaker == UNKNOWN_SPEAKER

    def test_empty_diarization(self):
        transcription = [TranscriptionSegment("hello", 0.0, 5.0)]
        result = merge([], transcription)
        assert result[0].speaker == UNKNOWN_SPEAKER

    def test_empty_transcription(self):
        diarization = [DiarizationSegment("A", 0.0, 5.0)]
        result = merge(diarization, [])
        assert result == []

    def test_both_empty(self):
        assert merge([], []) == []

    def test_multiple_speakers(self):
        diarization = [
            DiarizationSegment("A", 0.0, 5.0),
            DiarizationSegment("B", 5.0, 10.0),
        ]
        transcription = [
            TranscriptionSegment("first", 0.0, 4.0),
            TranscriptionSegment("second", 6.0, 9.0),
        ]
        result = merge(diarization, transcription)
        assert result[0].speaker == "A"
        assert result[1].speaker == "B"

    def test_preserves_timestamps(self):
        diarization = [DiarizationSegment("A", 0.0, 10.0)]
        transcription = [TranscriptionSegment("hi", 1.5, 3.7)]
        result = merge(diarization, transcription)
        assert result[0].start == 1.5
        assert result[0].end == 3.7


class TestFormatText:
    """Tests for format_text()."""

    def test_basic(self):
        segments = [MergedSegment("A", "hello world", 0.0, 2.0)]
        assert format_text(segments) == "A: hello world\n"

    def test_consecutive_same_speaker_merged(self):
        segments = [
            MergedSegment("A", "hello", 0.0, 1.0),
            MergedSegment("A", "world", 1.0, 2.0),
        ]
        assert format_text(segments) == "A: hello world\n"

    def test_speaker_change(self):
        segments = [
            MergedSegment("A", "hello", 0.0, 1.0),
            MergedSegment("B", "world", 1.0, 2.0),
        ]
        result = format_text(segments)
        assert "A: hello" in result
        assert "B: world" in result

    def test_blank_line_between_speakers(self):
        segments = [
            MergedSegment("A", "hello", 0.0, 1.0),
            MergedSegment("B", "world", 1.0, 2.0),
        ]
        result = format_text(segments)
        assert "\n\n" in result

    def test_empty(self):
        assert format_text([]) == ""


class TestFormatJson:
    """Tests for format_json()."""

    def test_basic_structure(self):
        segments = [MergedSegment("A", "hello", 0.531, 2.874)]
        data = json.loads(format_json(segments))
        assert data["speakers"] == ["A"]
        assert len(data["segments"]) == 1
        assert data["segments"][0]["speaker"] == "A"
        assert data["segments"][0]["text"] == "hello"
        assert data["segments"][0]["start"] == 0.531
        assert data["segments"][0]["end"] == 2.874

    def test_empty(self):
        data = json.loads(format_json([]))
        assert data["speakers"] == []
        assert data["segments"] == []

    def test_speakers_sorted(self):
        segments = [
            MergedSegment("B", "first", 0.0, 1.0),
            MergedSegment("A", "second", 1.0, 2.0),
        ]
        data = json.loads(format_json(segments))
        assert data["speakers"] == ["A", "B"]

    def test_timestamps_rounded(self):
        segments = [MergedSegment("A", "hi", 1.23456789, 2.98765432)]
        data = json.loads(format_json(segments))
        assert data["segments"][0]["start"] == 1.235
        assert data["segments"][0]["end"] == 2.988


class TestFormatTranscriptText:
    """Tests for format_transcript_text()."""

    def test_basic(self):
        segments = [TranscriptionSegment("hello world", 0.0, 2.0)]
        assert format_transcript_text(segments) == "hello world\n"

    def test_multiple_segments_joined(self):
        segments = [
            TranscriptionSegment("hello", 0.0, 1.0),
            TranscriptionSegment("world", 1.0, 2.0),
        ]
        assert format_transcript_text(segments) == "hello world\n"

    def test_empty(self):
        assert format_transcript_text([]) == ""


class TestFormatTranscriptJson:
    """Tests for format_transcript_json()."""

    def test_basic_structure(self):
        segments = [TranscriptionSegment("hello", 0.531, 2.874)]
        data = json.loads(format_transcript_json(segments))
        assert "speakers" not in data
        assert len(data["segments"]) == 1
        assert "speaker" not in data["segments"][0]
        assert data["segments"][0]["text"] == "hello"
        assert data["segments"][0]["start"] == 0.531
        assert data["segments"][0]["end"] == 2.874

    def test_empty(self):
        data = json.loads(format_transcript_json([]))
        assert data["segments"] == []

    def test_timestamps_rounded(self):
        segments = [TranscriptionSegment("hi", 1.23456789, 2.98765432)]
        data = json.loads(format_transcript_json(segments))
        assert data["segments"][0]["start"] == 1.235
        assert data["segments"][0]["end"] == 2.988


class TestFindFfmpeg:
    """Tests for _find_ffmpeg()."""

    def test_finds_ffmpeg(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/ffmpeg")
        assert _find_ffmpeg() == "/usr/local/bin/ffmpeg"

    def test_missing_ffmpeg_raises(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(ScribeError, match="ffmpeg is required"):
            _find_ffmpeg()


@pytest.fixture
def _mock_ffmpeg(monkeypatch):
    """Stub out ffmpeg discovery and subprocess.run for _prepare_audio tests."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 0),
    )


class TestPrepareAudio:
    """Tests for _prepare_audio()."""

    @pytest.mark.usefixtures("_mock_ffmpeg")
    def test_wav_still_converts(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.touch()
        with _prepare_audio(wav) as result:
            assert result.suffix == ".wav"
            assert result != wav

    @pytest.mark.usefixtures("_mock_ffmpeg")
    def test_non_wav_converts(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        with _prepare_audio(mp4) as result:
            assert result.suffix == ".wav"
            assert result != mp4

    @pytest.mark.usefixtures("_mock_ffmpeg")
    def test_temp_file_cleaned_up(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        temp_path = None
        with _prepare_audio(mp4) as result:
            temp_path = result
        assert not temp_path.exists()

    def test_ffmpeg_failure_raises(self, tmp_path, monkeypatch):
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        def fail_run(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, "ffmpeg", stderr="bad codec")

        monkeypatch.setattr("subprocess.run", fail_run)
        with pytest.raises(ScribeError, match="ffmpeg conversion failed"), _prepare_audio(mp4):
            pass


class TestCreateDiarizer:
    """Tests for _create_diarizer()."""

    def test_missing_senko_raises(self, monkeypatch):
        import builtins  # noqa: PLC0415

        real_import = builtins.__import__

        def block_senko(name, *args, **kwargs):
            if name == "senko":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_senko)
        with pytest.raises(ScribeError, match="senko is not installed"):
            _create_diarizer()

    def test_init_failure_raises(self, monkeypatch):
        mock_senko = MagicMock()
        mock_senko.Diarizer.side_effect = RuntimeError("CoreML init failed")
        monkeypatch.setitem(sys.modules, "senko", mock_senko)
        with pytest.raises(ScribeError, match="Failed to initialize diarizer"):
            _create_diarizer()


class TestCreateTranscriber:
    """Tests for _create_transcriber()."""

    def test_missing_nemo_raises(self, monkeypatch):
        import builtins  # noqa: PLC0415

        real_import = builtins.__import__

        def block_nemo(name, *args, **kwargs):
            if name == "nemo" or name.startswith("nemo."):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_nemo)
        with pytest.raises(ScribeError, match="nemo is not installed"):
            _create_transcriber()

    def test_init_failure_raises(self, monkeypatch):
        import types  # noqa: PLC0415

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        fake_nemo_asr = MagicMock()
        fake_nemo_asr.models.ASRModel.from_pretrained.side_effect = RuntimeError("download failed")
        fake_nemo = types.SimpleNamespace()
        fake_nemo.collections = types.SimpleNamespace(asr=fake_nemo_asr)
        nemo_pkg = types.ModuleType("nemo")
        nemo_pkg.collections = fake_nemo.collections
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "nemo", nemo_pkg)
        monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
        monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_nemo_asr)
        with pytest.raises(ScribeError, match="Failed to initialize transcriber"):
            _create_transcriber()

    def test_loads_to_cuda_and_sets_local_attention(self, monkeypatch):
        import types  # noqa: PLC0415

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        model = MagicMock()
        model.to.return_value = model
        asr_models = MagicMock()
        asr_models.models.ASRModel.from_pretrained.return_value = model
        nemo_pkg = types.ModuleType("nemo")
        nemo_pkg.collections = types.SimpleNamespace(asr=asr_models)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "nemo", nemo_pkg)
        monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
        monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_models)

        result = _create_transcriber()
        assert result is model
        asr_models.models.ASRModel.from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        model.to.assert_called_once_with("cuda")
        model.change_attention_model.assert_called_once_with(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256],
        )


class TestRunTranscription:
    """Tests for run_transcription()."""

    def _make_transcriber(self, segment_stamps):
        """Build a mock NeMo model whose transcribe() returns one hypothesis."""
        hyp = types.SimpleNamespace(timestamp={"segment": segment_stamps})
        transcriber = MagicMock()
        transcriber.transcribe.return_value = [hyp]
        return transcriber

    def test_segments_parsed(self, tmp_path):
        stamps = [{"segment": "hello", "start": 0.0, "end": 1.0}]
        result = run_transcription(self._make_transcriber(stamps), tmp_path / "audio.wav")
        assert len(result) == 1
        assert result[0] == TranscriptionSegment("hello", 0.0, 1.0)

    def test_multiple_segments(self, tmp_path):
        stamps = [
            {"segment": "hello", "start": 0.0, "end": 1.0},
            {"segment": "world", "start": 1.0, "end": 2.0},
        ]
        result = run_transcription(self._make_transcriber(stamps), tmp_path / "audio.wav")
        assert [s.text for s in result] == ["hello", "world"]

    def test_empty_segments(self, tmp_path):
        result = run_transcription(self._make_transcriber([]), tmp_path / "audio.wav")
        assert result == []

    def test_missing_fields_skipped(self, tmp_path):
        stamps = [
            {"segment": "ok"},
            {"segment": "hi", "start": 0.0, "end": 1.0},
        ]
        result = run_transcription(self._make_transcriber(stamps), tmp_path / "audio.wav")
        assert len(result) == 1
        assert result[0].text == "hi"

    def test_empty_text_skipped(self, tmp_path):
        stamps = [{"segment": "   ", "start": 0.0, "end": 1.0}]
        result = run_transcription(self._make_transcriber(stamps), tmp_path / "audio.wav")
        assert result == []

    def test_no_output_raises(self, tmp_path):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = []
        with pytest.raises(ScribeError, match="no transcription output"):
            run_transcription(transcriber, tmp_path / "audio.wav")

    def test_transcribe_error_wrapped(self, tmp_path):
        transcriber = MagicMock()
        transcriber.transcribe.side_effect = RuntimeError("boom")
        with pytest.raises(ScribeError, match="Transcription failed"):
            run_transcription(transcriber, tmp_path / "audio.wav")

    def test_missing_timestamp_attr_uses_empty(self, tmp_path):
        hyp = types.SimpleNamespace(timestamp=None)
        transcriber = MagicMock()
        transcriber.transcribe.return_value = [hyp]
        result = run_transcription(transcriber, tmp_path / "audio.wav")
        assert result == []


class TestRunDiarization:
    """Tests for run_diarization()."""

    def test_returns_segments(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = {
            "merged_segments": [
                {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.5},
                {"speaker": "SPEAKER_02", "start": 2.5, "end": 5.0},
            ],
        }
        result = run_diarization(diarizer, Path("/fake/audio.wav"))
        assert len(result) == 2
        assert result[0] == DiarizationSegment("SPEAKER_01", 0.0, 2.5)
        assert result[1] == DiarizationSegment("SPEAKER_02", 2.5, 5.0)

    def test_none_result_returns_empty(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = None
        result = run_diarization(diarizer, Path("/fake/audio.wav"))
        assert result == []

    def test_empty_segments_returns_empty(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = {"merged_segments": []}
        result = run_diarization(diarizer, Path("/fake/audio.wav"))
        assert result == []

    def test_malformed_result_raises(self):
        diarizer = MagicMock()
        diarizer.diarize.return_value = {"wrong_key": []}
        with pytest.raises(ScribeError, match="Diarization failed"):
            run_diarization(diarizer, Path("/fake/audio.wav"))

    def test_wraps_diarizer_errors(self):
        diarizer = MagicMock()
        diarizer.diarize.side_effect = RuntimeError("boom")
        with pytest.raises(ScribeError, match="Diarization failed"):
            run_diarization(diarizer, Path("/fake/audio.wav"))


class TestMain:
    """Tests for main() CLI entry point."""

    def test_file_not_found(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["scribe", "/nonexistent/audio.wav"])
        with pytest.raises(SystemExit, match="audio file not found"):
            main()

    def test_diarize_to_file(self, tmp_path, monkeypatch):
        audio = tmp_path / "meeting.wav"
        audio.touch()
        out = tmp_path / "meeting.txt"

        @contextmanager
        def fake_prepare(path):
            yield path

        monkeypatch.setattr("scribe._prepare_audio", fake_prepare)
        monkeypatch.setattr(
            "scribe._run_with_diarization",
            lambda *_a, **_kw: "SPEAKER_00: hello\n",
        )
        monkeypatch.setattr("sys.argv", ["scribe", str(audio), "-o", str(out)])
        main()
        assert out.read_text() == "SPEAKER_00: hello\n"

    def test_no_diarize_to_stdout(self, tmp_path, monkeypatch, capsys):
        audio = tmp_path / "meeting.wav"
        audio.touch()

        @contextmanager
        def fake_prepare(path):
            yield path

        monkeypatch.setattr("scribe._prepare_audio", fake_prepare)
        monkeypatch.setattr(
            "scribe._run_transcript_only",
            lambda *_a, **_kw: "hello world\n",
        )
        monkeypatch.setattr(
            "sys.argv",
            ["scribe", str(audio), "-o", "-", "--no-diarize"],
        )
        main()
        assert capsys.readouterr().out == "hello world\n"

    def test_default_output_path_json(self, tmp_path, monkeypatch):
        audio = tmp_path / "meeting.m4a"
        audio.touch()

        @contextmanager
        def fake_prepare(path):
            yield path

        monkeypatch.setattr("scribe._prepare_audio", fake_prepare)
        monkeypatch.setattr(
            "scribe._run_with_diarization",
            lambda *_a, **_kw: '{"speakers": []}\n',
        )
        monkeypatch.setattr(
            "sys.argv",
            ["scribe", str(audio), "--format", "json"],
        )
        main()
        expected = tmp_path / "meeting.json"
        assert expected.exists()
        assert expected.read_text() == '{"speakers": []}\n'

    def test_scribe_error_exits(self, tmp_path, monkeypatch):
        audio = tmp_path / "meeting.wav"
        audio.touch()

        @contextmanager
        def fake_prepare(path):
            yield path

        monkeypatch.setattr("scribe._prepare_audio", fake_prepare)
        monkeypatch.setattr(
            "scribe._run_with_diarization",
            MagicMock(side_effect=ScribeError("diarization broke")),
        )
        monkeypatch.setattr("sys.argv", ["scribe", str(audio)])
        with pytest.raises(SystemExit, match="diarization broke"):
            main()
