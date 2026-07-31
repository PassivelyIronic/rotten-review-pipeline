{% macro score_to_100(col) %}
    case
        when {{ col }} is null then null
        when regexp_matches(trim({{ col }}), '^\d+(\.\d+)?\s*%$')
            then cast(regexp_extract(trim({{ col }}), '^(\d+(?:\.\d+)?)', 1) as double)
        when regexp_matches(trim({{ col }}), '^\d+(\.\d+)?\s*/\s*\d+(\.\d+)?$')
            then 100.0 * cast(regexp_extract(trim({{ col }}), '^(\d+(?:\.\d+)?)', 1) as double)
                 / nullif(cast(regexp_extract(trim({{ col }}), '/\s*(\d+(?:\.\d+)?)$', 1) as double), 0)
        when regexp_matches(trim({{ col }}), '^\d+(\.\d+)?$') then
            case
                when cast(trim({{ col }}) as double) <= 5
                    then 100.0 * cast(trim({{ col }}) as double) / 5.0
                when cast(trim({{ col }}) as double) <= 10
                    then cast(trim({{ col }}) as double) * 10.0
                when cast(trim({{ col }}) as double) <= 100
                    then cast(trim({{ col }}) as double)
                else null
            end
        else
            case upper(trim({{ col }}))
                when 'A+' then 100 when 'A' then 95 when 'A-' then 90
                when 'B+' then 87  when 'B' then 83 when 'B-' then 80
                when 'C+' then 77  when 'C' then 73 when 'C-' then 70
                when 'D+' then 67  when 'D' then 63 when 'D-' then 60
                when 'F'  then 50
                else null
            end
    end
{% endmacro %}
