-- Burst ratio is a share: it must stay in (0, 1].
select review_id, reviewer_burst_ratio_7d
from {{ ref('int_burst_activity') }}
where reviewer_burst_ratio_7d <= 0 or reviewer_burst_ratio_7d > 1
