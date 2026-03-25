import json
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from typing import Sequence
from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)

class JsonLinesSpanExporter(SpanExporter):
    """
    Custom span exporter that serializes spans to a JSON Lines file.
    This allows easy inspection of LLM traces, context, and outputs.
    """
    def __init__(self, output_file="traces.jsonl"):
        self.output_file = output_file

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                for span in spans:
                    # Convert span data to dictionary for JSON serialization
                    span_dict = {
                        "name": span.name,
                        "context": {
                            "trace_id": format(span.get_span_context().trace_id, '032x'),
                            "span_id": format(span.get_span_context().span_id, '016x'),
                            "trace_flags": span.get_span_context().trace_flags,
                        },
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "status": {"status_code": span.status.status_code.name},
                        "attributes": dict(span.attributes) if span.attributes else {},
                        "events": [{"name": event.name, "timestamp": event.timestamp, "attributes": dict(event.attributes) if event.attributes else {}} for event in span.events],
                        "links": [{"context": {"trace_id": format(link.context.trace_id, '032x'), "span_id": format(link.context.span_id, '016x')}, "attributes": dict(link.attributes) if link.attributes else {}} for link in span.links],
                    }
                    f.write(json.dumps(span_dict, ensure_ascii=False) + "\\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans to JSONL: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def setup_telemetry(output_file="traces.jsonl"):
    """
    Configures the global OpenTelemetry TracerProvider to use the JsonLinesSpanExporter.
    AutoGen framework natively uses this global TracerProvider if configured.
    """
    provider = TracerProvider()
    
    # Use SimpleSpanProcessor to ensure spans are processed synchronously line by line
    exporter = JsonLinesSpanExporter(output_file=output_file)
    processor = SimpleSpanProcessor(exporter)
    
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    print(f"[*] OpenTelemetry abilitato. Tracce in streaming su: {output_file}")
