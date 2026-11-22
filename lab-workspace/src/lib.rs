#[derive(Debug, PartialEq, Eq)]
pub enum DurationError {
    Empty,
    InvalidNumber(String),
    UnknownUnit(String),
}

/// Parses human-friendly duration strings such as `"250ms"`, `"5s"`, or `"2m"`
/// into milliseconds.
pub fn parse_duration_ms(input: &str) -> Result<u64, DurationError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(DurationError::Empty);
    }
    if let Some(num) = input.strip_suffix("ms") {
        return parse_u64(num);
    }
    if let Some(num) = input.strip_suffix('s') {
        let secs = parse_u64(num)?;
        return Ok(secs * 1_000);
    }
    if let Some(num) = input.strip_suffix('m') {
        let mins = parse_u64(num)?;
        return Ok(mins * 60_000);
    }
    Err(DurationError::UnknownUnit(input.to_string()))
}

fn parse_u64(raw: &str) -> Result<u64, DurationError> {
    raw.parse::<u64>()
        .map_err(|_| DurationError::InvalidNumber(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seconds_and_minutes() {
        assert_eq!(parse_duration_ms("5s"), Ok(5_000));
        assert_eq!(parse_duration_ms("2m"), Ok(120_000));
    }

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(parse_duration_ms("  12s \n"), Ok(12_000));
    }

    #[test]
    fn parses_milliseconds() {
        assert_eq!(parse_duration_ms("250ms"), Ok(250));
    }

    #[test]
    fn rejects_empty_and_unknown_units() {
        assert_eq!(parse_duration_ms("   "), Err(DurationError::Empty));
        assert_eq!(
            parse_duration_ms("10h"),
            Err(DurationError::UnknownUnit("10h".to_string()))
        );
    }
}
