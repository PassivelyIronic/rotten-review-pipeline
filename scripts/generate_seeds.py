"""Generate the synthetic seed data used by CI and by the notebook's sample mode.

The seeds mirror the schema of the Kaggle dataset
`stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset` so the same
dbt models run against both. Everything here is generated, not scraped: no real
critic names, publications or review text.

Deliberately planted edge cases (asserted by tests/test_dbt_pipeline.py):
  * a critic publishing in tight bursts
  * a non-top-critic row that staging must filter out
  * an unparseable review score that must stay NULL rather than clamp to 1
  * scores in every supported format: "4/5", "8/10", "B+", "85%", "9"
  * text/score mismatches (glowing text with a low score and vice versa)
  * a share of reviews that spell their own verdict out in the body text
    ("Verdict: 3/5"), which is how the target leaks into text features on the
    real dataset too — see notebooks/01_review_integrity_analysis.ipynb
  * exact duplicate rows and rows with no critic name, both present in the real
    Kaggle dump and both stripped by staging before reviewer-level aggregates
    are computed
  * exact duplicate rows and rows with no critic name, both of which the real
    Kaggle dump contains and staging has to strip before reviewer-level
    aggregates mean anything

Run: python scripts/generate_seeds.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEEDS_DIR = Path(__file__).resolve().parents[1] / "dbt" / "seeds"
RNG = random.Random(20260731)

STUDIOS = ["Nova Films", "Orbit Studio", "Pinewheel", "Halcyon Pictures", "Meridian"]
PUBLISHERS = [
    "Daily Reel",
    "Cinema Post",
    "Frame Weekly",
    "The Reeling",
    "Northside Review",
    "Screen Ledger",
]
CRITICS = [
    "Anna Kowal",
    "Marco Reyes",
    "Priya Nair",
    "Tom Becker",
    "Lena Fischer",
    "Ivan Petrov",
    "Sofia Marchetti",
    "Daniel Osei",
    "Hannah Lindqvist",
    "Yusuf Demir",
    "Clara Benoit",
    "Ravi Menon",
    "Greta Halvorsen",
    "Oscar Delgado",
    "Mei Tanaka",
    "Julian Frost",
    "Nadia Haddad",
    "Peter Vance",
    "Rosa Iglesias",
    "Kwame Boateng",
]

TITLE_A = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Silent",
    "Crimson",
    "Northern",
    "Last",
    "Broken",
    "Golden",
    "Hollow",
]
TITLE_B = [
    "City",
    "Road",
    "Night",
    "Sea",
    "Sky",
    "Harvest",
    "Orbit",
    "Signal",
    "Winter",
    "Garden",
    "Machine",
    "Anthem",
]

POSITIVE_OPENERS = [
    "A sharp, moving film with terrific performances and real momentum",
    "Beautifully shot and confidently directed, it earns every beat",
    "Smart writing and a superb lead turn make this an easy recommendation",
    "An absolute delight from the first frame to the last",
    "Assured, generous filmmaking that trusts its audience",
    "A gorgeous, patient picture with a startling final act",
]
POSITIVE_CLAUSES = [
    "the ensemble is uniformly excellent",
    "the cinematography does real narrative work",
    "it finds humour without undercutting the drama",
    "the pacing never slackens",
    "the score is used sparingly and lands hard",
    "the script respects its own premise",
]
NEGATIVE_OPENERS = [
    "A tedious, overlong mess that squanders a promising premise",
    "Flat characters and clumsy pacing sink whatever goodwill it builds",
    "Predictable and lifeless, with dialogue that lands with a thud",
    "A disappointing misfire that never finds its footing",
    "Weightless spectacle in search of a reason to exist",
    "Handsomely mounted and entirely inert",
]
NEGATIVE_CLAUSES = [
    "the leads have no chemistry to speak of",
    "every twist arrives twenty minutes after you saw it coming",
    "the editing is actively confusing",
    "it mistakes volume for tension",
    "the third act abandons its own rules",
    "the humour is broad and joyless",
]
MIXED_CLAUSES = [
    "it works better as a mood piece than as a story",
    "the ambition is admirable even when the execution wobbles",
    "there is a leaner, stranger film buried in here",
]

POSITIVE_SCORES = ["4/5", "8/10", "B+", "A-", "85%", "9", "4.5/5", "3.5/4", "B"]
NEGATIVE_SCORES = ["1.5/5", "3/10", "D", "C-", "35%", "2", "1/4", "D+"]
MIDDLE_SCORES = ["3/5", "6/10", "C+", "60%", "5", "2.5/4"]


def _review_text(kind: str, index: int, score: str | None = None) -> str:
    if kind == "positive":
        parts = [RNG.choice(POSITIVE_OPENERS), RNG.choice(POSITIVE_CLAUSES)]
    elif kind == "negative":
        parts = [RNG.choice(NEGATIVE_OPENERS), RNG.choice(NEGATIVE_CLAUSES)]
    else:
        parts = [RNG.choice(POSITIVE_OPENERS), RNG.choice(MIXED_CLAUSES)]
    if RNG.random() < 0.35:
        parts.append(RNG.choice(POSITIVE_CLAUSES if kind == "positive" else NEGATIVE_CLAUSES))
    text = f"{parts[0]}, and {'; '.join(parts[1:])}. (Review {index})"
    # ~30% of critics restate the score in prose, exactly as they do in the
    # real dataset; this is what makes digits in the text a leakage risk.
    if score and RNG.random() < 0.30:
        text = f"{text} Verdict: {score}."
    return text


def build_movies(n: int = 40) -> list[list]:
    rows = []
    used = set()
    base = date(2019, 1, 4)
    while len(rows) < n:
        title = f"{RNG.choice(TITLE_A)} {RNG.choice(TITLE_B)}"
        if title in used:
            continue
        used.add(title)
        release = base + timedelta(days=RNG.randint(0, 1500))
        rows.append(
            [
                f"m/{title.lower().replace(' ', '_')}",
                title,
                release.isoformat(),
                RNG.randint(12, 98),
                RNG.choice(STUDIOS),
            ]
        )
    return rows


def build_reviews(movies: list[list]) -> list[list]:
    rows: list[list] = []
    index = 0

    for critic in CRITICS:
        publisher = RNG.choice(PUBLISHERS)
        # a critic's disposition: how often they write positively
        positivity = RNG.uniform(0.35, 0.8)
        for _ in range(RNG.randint(15, 35)):
            index += 1
            movie = RNG.choice(movies)
            release = date.fromisoformat(movie[2])
            review_date = release + timedelta(days=RNG.randint(-5, 400))
            roll = RNG.random()
            if roll < positivity:
                kind, score = "positive", RNG.choice(POSITIVE_SCORES)
            elif roll < positivity + 0.15:
                kind, score = "mixed", RNG.choice(MIDDLE_SCORES)
            else:
                kind, score = "negative", RNG.choice(NEGATIVE_SCORES)
            rows.append(
                [
                    movie[0],
                    critic,
                    True,
                    publisher,
                    "Fresh" if kind != "negative" else "Rotten",
                    score,
                    review_date.isoformat(),
                    _review_text(kind, index, score),
                ]
            )

    # planted signal 1: a critic publishing in tight bursts, always glowing
    burst_movies = RNG.sample(movies, 14)
    burst_start = date(2021, 6, 1)
    for offset, movie in enumerate(burst_movies):
        index += 1
        rows.append(
            [
                movie[0],
                "Burst Bot",
                True,
                "Daily Reel",
                "Fresh",
                "5/5",
                (burst_start + timedelta(days=offset // 3)).isoformat(),
                _review_text("positive", index),
            ]
        )

    # planted signal 2: text/score mismatches (glowing prose, punishing score)
    for movie in RNG.sample(movies, 8):
        index += 1
        rows.append(
            [
                movie[0],
                "Peter Vance",
                True,
                "Screen Ledger",
                "Rotten",
                "2/10",
                (date.fromisoformat(movie[2]) + timedelta(days=RNG.randint(0, 60))).isoformat(),
                _review_text("positive", index),
            ]
        )

    # edge cases the tests assert on
    first_movie = movies[0][0]
    rows.append(
        [
            first_movie,
            "Random User",
            False,
            "Blogspot",
            "Fresh",
            "5/5",
            "2021-03-02",
            "best movie ever, no notes",
        ]
    )
    rows.append(
        [
            movies[1][0],
            "Anna Kowal",
            True,
            "Cinema Post",
            "Rotten",
            "two thumbs down",
            "2021-06-01",
            _review_text("negative", index + 1),
        ]
    )
    rows.append(
        [
            movies[2][0],
            "Tom Becker",
            True,
            "Frame Weekly",
            "Fresh",
            "",
            "2021-02-11",
            _review_text("positive", index + 2),
        ]
    )

    # source-data pathologies that staging has to strip (see stg_rt_reviews.sql
    # and int_burst_activity.sql); all four are present in the real dump
    rows.extend([list(row) for row in RNG.sample(rows[:200], 12)])
    for row in RNG.sample(rows[:200], 6):
        nameless = list(row)
        nameless[1] = ""
        rows.append(nameless)

    # archive backfill: a critic's back catalogue imported under one timestamp.
    # Counted naively this reads as the largest "burst" in the dataset.
    for offset, movie in enumerate(RNG.choices(movies, k=40)):
        rows.append(
            [
                movie[0],
                "Archive Critic",
                True,
                "The Reeling",
                "Fresh",
                "7/10",
                "2015-06-01",
                _review_text("positive", 9000 + offset),
            ]
        )

    # sentinel publication dates: placeholders, not real dates
    for offset, movie in enumerate(RNG.choices(movies, k=5)):
        rows.append(
            [
                movie[0],
                "Mordaunt Placeholder",
                True,
                "Cinema Post",
                "Fresh",
                "8/10",
                "1800-01-01",
                _review_text("positive", 9500 + offset),
            ]
        )

    return rows


def build_imdb(n: int = 400) -> list[list]:
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append([_review_text("positive", i), "positive"])
        else:
            rows.append([_review_text("negative", i), "negative"])
    return rows


def main() -> None:
    movies = build_movies()
    reviews = build_reviews(movies)
    imdb = build_imdb()

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEEDS_DIR / "sample_rt_movies.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rotten_tomatoes_link",
                "movie_title",
                "original_release_date",
                "tomatometer_rating",
                "production_company",
            ]
        )
        writer.writerows(movies)

    with open(SEEDS_DIR / "sample_rt_reviews.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rotten_tomatoes_link",
                "critic_name",
                "top_critic",
                "publisher_name",
                "review_type",
                "review_score",
                "review_date",
                "review_content",
            ]
        )
        writer.writerows(reviews)

    with open(SEEDS_DIR / "sample_imdb_reviews.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["review", "sentiment"])
        writer.writerows(imdb)

    print(f"movies={len(movies)} reviews={len(reviews)} imdb={len(imdb)}")


if __name__ == "__main__":
    main()
