-- Burst detection: reviews by the same critic in a trailing 7-day window.
-- Replaces the O(n^2) pandas groupby-apply loop from the original notebook
-- with a single window function.
--
-- Archive backfill has to be excluded first. Rotten Tomatoes imported the back
-- catalogues of long-serving critics under a single timestamp, so one critic can
-- carry thousands of reviews sharing one date. A RANGE window includes every row
-- tied on the ordering value — correct SQL, wrong answer here — which made the
-- most prolific and most respected critics look like the most suspicious ones
-- (the first full run scored Roger Ebert at 2,159 "reviews in 7 days").
--
-- Rows in a backfill batch keep an is_backfill_batch flag and get NULL burst
-- features rather than an invented number: their publication cadence is simply
-- not recorded in this dataset.

with dated as (
    select
        review_id,
        critic_name,
        review_date,
        count(*) over (partition by critic_name, review_date) as reviews_same_day
    from {{ ref('stg_rt_reviews') }}
    where review_date is not null
),

classified as (
    select
        *,
        reviews_same_day > {{ var('backfill_same_day_threshold', 8) }} as is_backfill_batch
    from dated
),

windowed as (
    select
        review_id,
        critic_name,
        reviews_same_day,
        count(*) over (
            partition by critic_name
            order by review_date
            range between interval 7 day preceding and current row
        ) as reviewer_reviews_last_7d,
        count(*) over (partition by critic_name) as reviewer_total_reviews
    from classified
    where not is_backfill_batch
)

select
    c.review_id,
    c.reviews_same_day,
    c.is_backfill_batch,
    w.reviewer_reviews_last_7d,
    w.reviewer_reviews_last_7d * 1.0 / w.reviewer_total_reviews as reviewer_burst_ratio_7d,
    w.reviewer_reviews_last_7d > {{ var('burst_review_threshold', 10) }} as is_reviewer_burst_7d,
    (w.reviewer_reviews_last_7d * 1.0 / w.reviewer_total_reviews)
        > {{ var('burst_ratio_threshold', 0.3) }} as is_reviewer_burst_normalized,
    coalesce(
        w.reviewer_reviews_last_7d > {{ var('burst_review_threshold', 10) }}
        or (w.reviewer_reviews_last_7d * 1.0 / w.reviewer_total_reviews)
            > {{ var('burst_ratio_threshold', 0.3) }},
        false
    ) as is_reviewer_burst_any
from classified c
left join windowed w using (review_id)
