import pandas as pd
from openpyxl import load_workbook

from scripts.update_public_reports import build_forward_summary, load_telegram_delivery_events, write_workbook


def test_forward_summary_groups_trade_results():
    trades = pd.DataFrame(
        [
            {"entry_type": "Standard Limit", "timeframe": "M30", "strategy": "FVG", "outcome": "WIN", "net_profit_usd": 3.0},
            {"entry_type": "Standard Limit", "timeframe": "M30", "strategy": "FVG", "outcome": "LOSS", "net_profit_usd": -1.0},
        ]
    )

    summary = build_forward_summary(trades)

    assert summary.to_dict("records") == [
        {
            "entry_type": "Standard Limit",
            "timeframe": "M30",
            "strategy": "FVG",
            "trades": 2,
            "wins": 1,
            "losses": 1,
            "winrate_pct": 50.0,
            "net_profit_usd": 2.0,
        }
    ]


def test_public_report_workbook_contains_expected_sheets(tmp_path):
    standard_limit = pd.DataFrame([{"timeframe": "M30", "strategy": "FVG", "winrate": 72.2}])
    trades = pd.DataFrame([{"position_id": 1, "outcome": "WIN", "net_profit_usd": 3.0}])
    summary = pd.DataFrame([{"entry_type": "Standard Limit", "trades": 1, "winrate_pct": 100.0}])
    output = tmp_path / "forward_test_report.xlsx"

    write_workbook(output, standard_limit, trades, summary, pd.DataFrame(), "2026-07-16 12:00:00 WIB")

    workbook = load_workbook(output, read_only=True)
    assert workbook.sheetnames == ["Overview", "Standard Limit Real Tick", "Forward Trades", "Forward Summary", "Telegram Delivery"]


def test_telegram_delivery_events_reads_valid_json_lines(tmp_path):
    path = tmp_path / "telegram_delivery_events.jsonl"
    path.write_text('{"sent_at_wib":"2026-07-16T12:00:00+07:00","channel":"text","delivered":true,"message":"entry"}\ninvalid\n', encoding="utf-8")

    events = load_telegram_delivery_events(path)

    assert events.to_dict("records") == [{"sent_at_wib": "2026-07-16T12:00:00+07:00", "channel": "text", "delivered": True, "message": "entry"}]
