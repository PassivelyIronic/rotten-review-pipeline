-- Rotten Tomatoes critic reviews (Clapper dataset), top critics only.
-- Parses free-form scores ("3/5", "B+", "85%", "7.5") to 0-100 and 1-10 scales.

with source as (
    {% if target.name in ('ci', 'test') %}
    select * from {{ ref('sample_rt_reviews') }}
    {% else %}
    select * from {{ source('raw', 'raw_rt_reviews') }}
    {% endif %}
),

filtered as (
    select *
    from source
    where top_critic = true
      and review_content is not null
      and trim(review_content) <> ''
),

parsed as (
    select
        md5(concat_ws('|', rotten_tomatoes_link, critic_name, review_date, review_content))
            as review_id,
        rotten_tomatoes_link,
        critic_name,
        publisher_name,
        cast(review_date as date) as review_date,
        review_content as review_text,
        length(review_content) - length(replace(review_content, ' ', '')) + 1 as word_count,
        case review_type when 'Fresh' then 1 when 'Rotten' then 0 end as review_type_encoded,
        {{ score_to_100('review_score') }} as score_100
    from filtered
),

split_assigned as (
    select
        *,
        -- deterministic 80/20 fold, keyed on review_id so a review always
        -- lands in the same fold no matter when the model is rebuilt
        case when abs(hash(review_id)) % 10 < 8 then 'train' else 'holdout' end as split
    from parsed
)

select
    *,
    -- guard: greatest/least in DuckDB skip NULLs, which would turn an
    -- unparseable score into 1 instead of keeping it NULL
    case
        when score_100 is null then null
        else cast(least(greatest(round(score_100 / 10.0), 1), 10) as integer)
    end as score_10
from split_assigned
