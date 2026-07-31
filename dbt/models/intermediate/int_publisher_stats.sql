-- Per-publisher score distribution (deviation-from-outlet signal).

select
    publisher_name,
    avg(score_10) as publisher_mean_score_10,
    stddev_samp(score_10) as publisher_std_score_10
from {{ ref('stg_rt_reviews') }}
where publisher_name is not null
group by publisher_name
