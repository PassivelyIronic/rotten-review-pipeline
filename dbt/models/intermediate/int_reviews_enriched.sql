-- Reviews joined with movie metadata + time-based features.

select
    r.*,
    m.movie_title,
    m.original_release_date,
    m.tomatometer_rating,
    m.production_company,
    date_diff('day', m.original_release_date, r.review_date) as days_since_release,
    date_diff('day', m.original_release_date, r.review_date) < 7 as is_early_review
from {{ ref('stg_rt_reviews') }} r
left join {{ ref('stg_rt_movies') }} m using (rotten_tomatoes_link)
