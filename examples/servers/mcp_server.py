from mcp.server.fastmcp import FastMCP
from typing import Annotated, List


mcp = FastMCP("troubleshooting")


@mcp.tool("event")
def get_event(eventId: Annotated[int, "event id"]) -> dict:
    """Get Event Detail"""
    return {
        "event": {
            "id": 10000,
            "version": 1,
            "date": {"start_time": "1700000000", "end_time": "1700010000"},
        }
    }


@mcp.tool("metric")
def get_metric(
    component: Annotated[str, "component name"],
    startTime: Annotated[int, "start time"],
    endTime: Annotated[int, "end time"],
) -> List:
    """Get metric data from prometheus by component name"""
    return [
        {
            "component": "IE",
            "name": "CPU Usage",
            "desc": "the cpu usage of the component, unit is %",
            "data": [[123456789, 10], [123456789, 12]],
        },
        {
            "component": "IE",
            "name": "Memory Usage",
            "desc": "the memory usage of the component, unit is %",
            "data": [[123456789, 10], [123456789, 12]],
        },
    ]


@mcp.tool("log")
def get_log(
    component: Annotated[str, "component name"],
    eventId: Annotated[int, "event id"],
    startTime: Annotated[int, "start time"],
    endTime: Annotated[int, "end time"],
) -> dict:
    """Get log from kibana by component name and event id"""
    return {
        "trace_id": "123456-123456-123456",
        "component": "IE",
        "event_id": "10000",
        "log": "sql execute failed",
    }


@mcp.tool("trace")
def get_trace(trace_id: Annotated[str, "trace id"]) -> List:
    """Get trace data by trace id"""
    return [
        {
            "eventId": 10000,
            "traceId": "123456-123456-123456",
            "process": "sql execute failed, no table exist: select * from schema.table",
        }
    ]


if __name__ == "__main__":
    mcp.run(transport='streamable-http')