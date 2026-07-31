-- Burst detection: reviews by the same critic in a trailing 7-day window.
-- Replaces the O(n^2) pandas groupby-apply loop from the original notebook
-- with a single window function.

with windowed as (
    select
        review_id,
        critic_name,
        review_date,
        count(*) over (
            partition by critic_name
            order by review_date
            range between interval 7 day preceding and current row
        ) as reviewer_reviews_last_7d,
        count(*) over (partition by critic_name) as reviewer_total_reviews
    from {{ ref('stg_rt_reviews') }}
    where review_date is not null
)

select
    review_id,
    reviewer_reviews_last_7d,
    reviewer_reviews_last_7d * 1.0 / reviewer_total_reviews as reviewer_burst_ratio_7d,
    reviewer_reviews_last_7d > 10 as is_reviewer_burst_7d,
    (reviewer_reviews_last_7d * 1.0 / reviewer_total_reviews) > 0.3
        as is_reviewer_burst_normalized,
    (reviewer_reviews_last_7d > 10)
        or ((reviewer_reviews_last_7d * 1.0 / reviewer_total_reviews) > 0.3)
        as is_reviewer_burst_any
from windowed
