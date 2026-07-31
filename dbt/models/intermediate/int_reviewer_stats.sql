-- Per-critic aggregates used for z-scores and the reviewer dimension.

select
    critic_name,
    count(*) as review_count,
    avg(review_type_encoded) as positive_ratio,
    avg(score_10) as critic_mean_score_10,
    stddev_samp(score_10) as critic_std_score_10,
    count(distinct rotten_tomatoes_link) as unique_movies,
    date_diff('day', min(review_date), max(review_date)) as days_active
from {{ ref('stg_rt_reviews') }}
group by critic_name
