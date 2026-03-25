# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.actions.lua_parser."""

from tritium_lib.actions.lua_parser import (
    MotorOutput,
    extract_lua_from_response,
    parse_function_call,
    split_arguments,
    parse_lua_value,
    parse_motor_output,
    format_motor_output,
)


def test_parse_function_call_basic():
    """parse_function_call parses simple function calls."""
    result = parse_function_call('say("Hello world")')
    assert result is not None
    name, args = result
    assert name == "say"
    assert args == ["Hello world"]


def test_parse_function_call_patrol():
    """parse_function_call handles patrol with nested JSON."""
    result = parse_function_call('patrol("unit_1", "[[1,2],[3,4]]")')
    assert result is not None
    name, args = result
    assert name == "patrol"
    assert args[0] == "unit_1"
    assert args[1] == "[[1,2],[3,4]]"


def test_parse_function_call_no_args():
    """parse_function_call handles zero-arg calls."""
    result = parse_function_call("scan()")
    assert result is not None
    assert result == ("scan", [])


def test_parse_function_call_numbers():
    """parse_function_call handles numeric arguments."""
    result = parse_function_call('dispatch("unit_1", 5.0, -3.5)')
    assert result is not None
    name, args = result
    assert name == "dispatch"
    assert args == ["unit_1", 5.0, -3.5]


def test_split_arguments():
    """split_arguments handles quoted strings with commas."""
    result = split_arguments('"hello, world", 42')
    assert len(result) == 2
    assert result[0] == '"hello, world"'
    assert result[1] == '42'


def test_parse_lua_value_string():
    """parse_lua_value handles strings."""
    assert parse_lua_value('"hello"') == "hello"
    assert parse_lua_value("'world'") == "world"


def test_parse_lua_value_number():
    """parse_lua_value handles numbers."""
    assert parse_lua_value("42") == 42
    assert parse_lua_value("3.14") == 3.14


def test_parse_lua_value_boolean():
    """parse_lua_value handles booleans."""
    assert parse_lua_value("true") is True
    assert parse_lua_value("false") is False


def test_parse_lua_value_nil():
    """parse_lua_value handles nil."""
    assert parse_lua_value("nil") is None


def test_extract_lua_from_response_code_block():
    """extract_lua_from_response extracts from code blocks."""
    response = '```lua\nsay("Hello")\n```'
    assert extract_lua_from_response(response) == 'say("Hello")'


def test_extract_lua_from_response_bare():
    """extract_lua_from_response finds bare function calls."""
    response = 'I think I should say("Hello") to greet them'
    result = extract_lua_from_response(response)
    assert 'say("Hello")' in result


def test_extract_lua_strips_think_tags():
    """extract_lua_from_response strips <think> tags."""
    response = '<think>reasoning here</think>say("Hello")'
    result = extract_lua_from_response(response)
    assert "think>" not in result.lower()


def test_motor_output_dataclass():
    """MotorOutput can be created with defaults."""
    m = MotorOutput()
    assert m.action == ""
    assert m.params == []
    assert m.valid is False


def test_parse_motor_output_valid():
    """parse_motor_output parses valid say() call."""
    result = parse_motor_output('say("Hello!")')
    assert result.valid is True
    assert result.action == "say"
    assert result.params == ["Hello!"]


def test_parse_motor_output_empty():
    """parse_motor_output handles empty input."""
    result = parse_motor_output("")
    assert result.valid is False
    assert result.error is not None


def test_format_motor_output():
    """format_motor_output formats valid output."""
    m = MotorOutput(action="say", params=["Hello"], valid=True)
    assert format_motor_output(m) == "say('Hello')"
