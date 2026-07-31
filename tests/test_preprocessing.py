from rotten_review.preprocessing import basic_text_stats, clean_text


def test_clean_text_strips_html_urls_punctuation_digits():
    raw = "<p>Great movie!!! 10/10 see https://example.com NOW.</p>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned and ">" not in cleaned
    assert "http" not in cleaned
    assert not any(ch.isdigit() for ch in cleaned)
    assert cleaned == cleaned.lower()
    assert "great movie" in cleaned


def test_clean_text_handles_none_and_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text("   ") == ""


def test_basic_text_stats():
    stats = basic_text_stats("great great movie")
    assert stats["word_count"] == 3
    assert 0 < stats["lexical_diversity"] < 1
    assert basic_text_stats("")["word_count"] == 0


def test_app_feature_row_matches_detector_contract():
    """The Gradio demo must build exactly the features the detector expects."""
    from rotten_review.app import _behavioural_row
    from rotten_review.config import ANOMALY

    context = {
        "reviews_last_7d": 3,
        "burst_ratio": 0.1,
        "positive_ratio": 0.7,
        "critic_std": 1.5,
        "days_since_release": 10,
        "z_score_critic": 0.2,
        "consensus_diff": 1.0,
        "publisher_z_score": 0.3,
    }
    row = _behavioural_row(context, word_count=40, score_residual=-2.0)
    assert list(row.columns) == list(ANOMALY.behavioural_features)
    assert len(row) == 1
    assert row.notna().all().all()
