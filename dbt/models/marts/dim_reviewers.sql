-- Reviewer dimension: activity profile + burst summary per critic.

with bursts as (
    select
        r.critic_name,
        sum(case when b.is_reviewer_burst_any then 1 else 0 end) as burst_review_count,
        max(b.reviewer_reviews_last_7d) as max_reviews_in_7d
    from {{ ref('stg_rt_reviews') }} r
    join {{ ref('int_burst_activity') }} b using (review_id)
    where not b.is_backfill_batch
    group by r.critic_name
)

select
    rs.*,
    coalesce(b.burst_review_count, 0) as burst_review_count,
    coalesce(b.max_reviews_in_7d, 0) as max_reviews_in_7d,
    coalesce(b.burst_review_count, 0) * 1.0 / rs.review_count as burst_share
from {{ ref('int_reviewer_stats') }} rs
left join bursts b using (critic_name)
