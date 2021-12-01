from google.protobuf import json_format

import mjx
import mjxproto
from mjx.const import EventType


def test_Event():
    json_str = '{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3}'
    event = mjx.Event(json_str)
    restored = event.to_json()
    assert json_str == restored


def test_to_json():
    json_str = '{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3}'
    event = mjx.Event(json_str)
    restored = event.to_json()
    assert json_str == restored


def test_to_proto():
    json_str = '{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3}'
    event = mjx.Event(json_str)
    event.to_proto() == json_format.Parse(json_str, mjxproto.Event())


def test_from_proto():
    json_str = '{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3}'
    event = mjx.Event(json_str)
    proto_event = event.to_proto()
    mjx.Event.from_proto(proto_event).to_json() == json_str


def test_type():
    json_str = '{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3}'
    event = mjx.Event(json_str)
    assert event.type() == 1
    assert event.type() == EventType.TSUMOGIRI
    assert event.type() != 0
    assert event.type() != EventType.DISCARD


def test_tile():
    json_str = '{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3}'
    event = mjx.Event(json_str)
    tile = event.tile()
    assert tile is not None
    assert tile.id() == 3

    json_str = '{"type":"EVENT_TYPE_DRAW","who":1}'
    event = mjx.Event(json_str)
    tile = event.tile()
    assert tile is None
