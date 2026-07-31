-- Parsed scores must land on the 1-10 scale.
select review_id, score_10
from {{ ref('stg_rt_reviews') }}
where score_10 is not null and (score_10 < 1 or score_10 > 10)
