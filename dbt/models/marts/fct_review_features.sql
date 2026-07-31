-- One row per review with every behavioural feature the anomaly model needs.
-- Text-model-dependent features (predicted sentiment/score, residuals,
-- composite bias score) are added in Python on top of this mart.

with base as (
    select
        e.review_id,
        e.rotten_tomatoes_link,
        e.movie_title,
        e.critic_name,
        e.publisher_name,
        e.production_company,
        e.review_date,
        e.split,
        e.review_text,
        e.word_count,
        e.review_type_encoded,
        e.score_100,
        e.score_10,
        e.tomatometer_rating,
        e.days_since_release,
        e.is_early_review,
        b.reviews_same_day,
        b.is_backfill_batch,
        b.reviewer_reviews_last_7d,
        b.reviewer_burst_ratio_7d,
        b.is_reviewer_burst_any,
        rs.review_count,
        rs.positive_ratio,
        rs.critic_mean_score_10,
        greatest(coalesce(rs.critic_std_score_10, 0.5), 0.5) as critic_std_score_10,
        rs.unique_movies,
        rs.days_active,
        ps.publisher_mean_score_10,
        greatest(coalesce(ps.publisher_std_score_10, 0.5), 0.5) as publisher_std_score_10
    from {{ ref('int_reviews_enriched') }} e
    left join {{ ref('int_burst_activity') }} b using (review_id)
    left join {{ ref('int_reviewer_stats') }} rs using (critic_name)
    left join {{ ref('int_publisher_stats') }} ps using (publisher_name)
)

select
    *,
    (score_10 - critic_mean_score_10) / critic_std_score_10 as z_score_critic,
    least(greatest(tomatometer_rating / 10.0, 0), 10) as consensus_score_10,
    abs(score_10 - least(greatest(tomatometer_rating / 10.0, 0), 10)) as consensus_diff,
    abs((score_10 - publisher_mean_score_10) / publisher_std_score_10) as publisher_z_score
from base
