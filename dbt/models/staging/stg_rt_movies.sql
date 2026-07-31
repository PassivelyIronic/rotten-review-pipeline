-- Movie metadata: release date, tomatometer consensus, production company.

with source as (
    {% if target.name in ('ci', 'test') %}
    select * from {{ ref('sample_rt_movies') }}
    {% else %}
    select * from {{ source('raw', 'raw_rt_movies') }}
    {% endif %}
)

select
    rotten_tomatoes_link,
    movie_title,
    cast(original_release_date as date) as original_release_date,
    cast(tomatometer_rating as double) as tomatometer_rating,
    production_company
from source
