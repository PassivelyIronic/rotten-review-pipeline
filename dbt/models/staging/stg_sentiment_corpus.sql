-- Training corpus for Model 1: IMDB 50k + Rotten Tomatoes top-critic reviews,
-- unified to (text, label) with label 1 = positive/Fresh, -1 = negative/Rotten.
--
-- The `split` column carries the RT fold assignment through, so the sentiment
-- model can be trained without ever seeing the reviews it is later evaluated
-- on. IMDB rows are training-only by construction.

with imdb as (
    {% if target.name in ('ci', 'test') %}
    select review, sentiment from {{ ref('sample_imdb_reviews') }}
    {% else %}
    select review, sentiment from {{ source('raw', 'raw_imdb_reviews') }}
    {% endif %}
),

imdb_labelled as (
    select
        review as text,
        case sentiment when 'positive' then 1 when 'negative' then -1 end as label,
        'imdb' as source,
        'train' as split
    from imdb
    where review is not null
),

rt as (
    select
        review_text as text,
        case review_type_encoded when 1 then 1 when 0 then -1 end as label,
        'rotten_tomatoes' as source,
        split
    from {{ ref('stg_rt_reviews') }}
    where review_type_encoded is not null
)

select * from imdb_labelled
union all
select * from rt
