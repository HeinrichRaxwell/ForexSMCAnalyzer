import os
import sys
import pandas as pd
import numpy as np
import pytest
import joblib
import json
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import (
    predict_setup_probability,
    update_feedback_data,
    trigger_auto_retrain,
    check_and_trigger_retraining,
    analyze_trade_outcome_reason,
    process_mt5_history_feedback,
)
from src.model_trainer import train_xgboost_filter

def test_predict_setup_probability(tmp_path):
    # Train a dummy model first
    np.random.seed(42)
    data = {
        'time': pd.date_range(start="2026-06-01", periods=10, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'hour': np.random.randint(0, 24, size=10),
        'day_of_week': np.random.randint(0, 7, size=10),
        'setup_type': np.random.randint(0, 2, size=10),
        'direction': np.random.choice([-1, 1], size=10),
        'entry_price': np.random.uniform(2300, 2350, size=10),
        'sl_price': np.random.uniform(2300, 2350, size=10),
        'tp_price': np.random.uniform(2300, 2350, size=10),
        'risk_pips': np.random.uniform(1.0, 10.0, size=10),
        'atr_14': np.random.uniform(1.0, 5.0, size=10),
        'trend': np.random.choice([-1, 1], size=10),
        'killzone': np.random.randint(0, 4, size=10),
        'label': np.random.choice([0, 1], size=10)
    }
    # Force some variation to avoid issues during stratification
    data['label'][:5] = 0
    data['label'][5:] = 1
    
    df = pd.DataFrame(data)
    labeled_data_path = tmp_path / "labeled_setups.csv"
    df.to_csv(labeled_data_path, index=False)
    
    model_dir = tmp_path / "models"
    train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))
    
    model_path = model_dir / "smc_xgb_classifier.joblib"
    
    # Test predict_setup_probability with single dict
    single_setup = {
        'hour': 8,
        'day_of_week': 2,
        'setup_type': 1,
        'direction': 1,
        'entry_price': 2315.5,
        'sl_price': 2310.0,
        'tp_price': 2326.5,
        'risk_pips': 5.5,
        'atr_14': 2.3,
        'trend': 1,
        'killzone': 0
    }
    
    prob = predict_setup_probability(single_setup, model_path=str(model_path))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    
    # Test predict_setup_probability with list of dicts
    multiple_setups = [single_setup, single_setup]
    probs = predict_setup_probability(multiple_setups, model_path=str(model_path))
    assert isinstance(probs, list)
    assert len(probs) == 2
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probs)

def test_update_feedback_data(tmp_path):
    labeled_data_path = tmp_path / "test_feedback.csv"
    
    trade = {
        'time': '2026-06-04 12:00:00',
        'hour': 12,
        'day_of_week': 3,
        'setup_type': 0,
        'direction': -1,
        'entry_price': 2345.0,
        'sl_price': 2348.0,
        'tp_price': 2339.0,
        'risk_pips': 3.0,
        'atr_14': 1.8,
        'trend': -1,
        'killzone': 2,
        'label': 1
    }
    
    # Update first time (creates file)
    update_feedback_data(trade, labeled_data_path=str(labeled_data_path))
    assert os.path.exists(labeled_data_path)
    
    df = pd.read_csv(labeled_data_path)
    assert len(df) == 1
    assert list(df.columns) == [
        'time', 'timeframe', 'hour', 'day_of_week', 'setup_type', 'direction', 
        'entry_price', 'sl_price', 'tp_price', 'risk_pips', 'atr_14', 
        'trend', 'relative_risk', 'killzone', 'fvg_width', 'relative_fvg_width',
        'near_psychological_level', 'knn_prob_sig', 'knn_prob_opp', 
        'dist_entry_to_poc', 'dist_entry_to_nearest_poc', 
        'dist_entry_to_pp', 'dist_entry_to_nearest_pivot',
        'floop_signal', 'floop_strength', 'floop_trend', 'floop_trend_aligned', 'pnl_relative', 'label'
    ]
    assert df.iloc[0]['hour'] == 12
    assert df.iloc[0]['label'] == 1
    
    # Update second time (appends file)
    trades_list = [trade, trade]
    update_feedback_data(trades_list, labeled_data_path=str(labeled_data_path))
    df_updated = pd.read_csv(labeled_data_path)
    assert len(df_updated) == 3


def test_update_feedback_data_aligns_to_existing_csv_column_order(tmp_path):
    labeled_data_path = tmp_path / "existing_feedback.csv"
    existing_columns = [
        'time', 'timeframe', 'hour', 'day_of_week', 'setup_type', 'direction',
        'entry_price', 'sl_price', 'tp_price', 'risk_pips', 'atr_14',
        'trend', 'relative_risk', 'killzone', 'fvg_width', 'relative_fvg_width',
        'floop_signal', 'floop_strength', 'dist_entry_to_pp',
        'dist_entry_to_nearest_pivot', 'label', 'near_psychological_level',
        'pnl_relative', 'floop_trend', 'floop_trend_aligned', 'knn_prob_sig',
        'knn_prob_opp', 'dist_entry_to_poc', 'dist_entry_to_nearest_poc'
    ]
    pd.DataFrame([{
        'time': '2026-06-01 00:00:00',
        'timeframe': 60,
        'hour': 0,
        'day_of_week': 0,
        'setup_type': 1,
        'direction': 1,
        'entry_price': 2300.0,
        'sl_price': 2295.0,
        'tp_price': 2310.0,
        'risk_pips': 5.0,
        'atr_14': 8.0,
        'trend': 1,
        'relative_risk': 0.625,
        'killzone': 0,
        'fvg_width': 0.0,
        'relative_fvg_width': 0.0,
        'floop_signal': 0,
        'floop_strength': 5.0,
        'dist_entry_to_pp': 0.0,
        'dist_entry_to_nearest_pivot': 0.0,
        'label': 1,
        'near_psychological_level': 1,
        'pnl_relative': 2.0,
        'floop_trend': 1,
        'floop_trend_aligned': 1,
        'knn_prob_sig': 0.7,
        'knn_prob_opp': 0.3,
        'dist_entry_to_poc': 0.0,
        'dist_entry_to_nearest_poc': 0.0,
    }], columns=existing_columns).to_csv(labeled_data_path, index=False)

    update_feedback_data({
        'time': '2026-06-05 20:01:12',
        'timeframe': 30,
        'hour': 20,
        'day_of_week': 4,
        'setup_type': 0,
        'direction': 1,
        'entry_price': 4403.183242,
        'sl_price': 4398.183242,
        'tp_price': 4413.183242,
        'risk_pips': 5.0,
        'atr_14': 8.0,
        'trend': 1,
        'relative_risk': 0.625,
        'killzone': 0,
        'fvg_width': 1.0,
        'relative_fvg_width': 0.125,
        'near_psychological_level': 1,
        'knn_prob_sig': 0.7,
        'knn_prob_opp': 0.3,
        'dist_entry_to_poc': 0.0,
        'dist_entry_to_nearest_poc': 0.0,
        'dist_entry_to_pp': 0.0,
        'dist_entry_to_nearest_pivot': 0.0,
        'floop_signal': 0,
        'floop_strength': 5.0,
        'floop_trend': 1,
        'floop_trend_aligned': 1,
        'pnl_relative': -1.0,
        'label': 0,
    }, labeled_data_path=str(labeled_data_path))

    df = pd.read_csv(labeled_data_path)
    assert list(df.columns) == existing_columns
    assert df.iloc[-1]['label'] == 0
    assert df.iloc[-1]['pnl_relative'] == -1.0
    assert df.iloc[-1]['near_psychological_level'] == 1
    assert df.iloc[-1]['dist_entry_to_nearest_poc'] == 0.0

def test_trigger_auto_retrain():
    with patch('src.model_trainer.train_xgboost_filter') as mock_train:
        trigger_auto_retrain()
        mock_train.assert_called_once()


def test_check_and_trigger_retraining_retrains_after_one_trade(tmp_path, monkeypatch):
    status_file = tmp_path / "learning_status.json"
    monkeypatch.setenv("ML_RETRAIN_THRESHOLD", "1")

    with patch("src.inference.trigger_auto_retrain", return_value=(object(), None)) as mock_train:
        result = check_and_trigger_retraining(1, status_file=str(status_file))

    mock_train.assert_called_once()
    assert result["retrained"] is True
    assert result["new_trades_since_last_train"] == 0


def test_check_and_trigger_retraining_alert_describes_holdout_metrics(tmp_path, monkeypatch):
    status_file = tmp_path / "learning_status.json"
    monkeypatch.setenv("ML_RETRAIN_THRESHOLD", "1")
    stats = {
        "status": "REJECTED",
        "dataset_size": 1219,
        "test_size": 244,
        "eval_threshold": 0.50,
        "old_accuracy": 81.56,
        "new_accuracy": 74.59,
        "old_winrate": 92.19,
        "new_winrate": 68.69,
        "old_passed_count": 128,
        "new_passed_count": 99,
    }

    with patch("src.inference.trigger_auto_retrain", return_value=(object(), stats)), \
         patch("src.model_trainer.get_training_max_setups", return_value=5000), \
         patch("src.telegram_bot.send_telegram_alert") as mock_alert:
        result = check_and_trigger_retraining(1, status_file=str(status_file))

    assert result["retrained"] is True
    message = mock_alert.call_args.args[0]
    assert "Basis Metric" in message
    assert "Holdout test 20%" in message
    assert "threshold 0.50" in message
    assert "Champion Test Winrate @ 0.50" in message
    assert "(128/244 lolos)" in message
    assert "bukan jaminan winrate live berikutnya" in message


def test_check_and_trigger_retraining_can_wait_for_five_trades(tmp_path, monkeypatch):
    status_file = tmp_path / "learning_status.json"
    monkeypatch.setenv("ML_RETRAIN_THRESHOLD", "5")
    monkeypatch.setenv("ML_RETRAIN_ON_WEEKEND", "false")

    with patch("src.inference.trigger_auto_retrain") as mock_train:
        result = check_and_trigger_retraining(1, status_file=str(status_file))

    mock_train.assert_not_called()
    assert result["retrained"] is False
    assert result["new_trades_since_last_train"] == 1


def test_process_mt5_history_feedback_backfills_recorded_trade_missing_from_csv(tmp_path, monkeypatch):
    sent_signals_path = tmp_path / "sent_signals.json"
    labeled_data_path = tmp_path / "labeled_setups.csv"
    signal_time = "2026-06-05 18:59:39"
    ticket_id = 123456
    features = {
        "timeframe": 15,
        "hour": 18,
        "day_of_week": 4,
        "setup_type": 0,
        "direction": 1,
        "entry_price": 2300.0,
        "sl_price": 2295.0,
        "tp_price": 2310.0,
        "risk_pips": 5.0,
        "atr_14": 8.0,
        "trend": 1,
        "relative_risk": 0.625,
        "killzone": 0,
        "fvg_width": 1.0,
        "relative_fvg_width": 0.125,
        "near_psychological_level": 1,
        "knn_prob_sig": 0.7,
        "knn_prob_opp": 0.3,
        "dist_entry_to_poc": 0.0,
        "dist_entry_to_nearest_poc": 0.0,
        "dist_entry_to_pp": 0.0,
        "dist_entry_to_nearest_pivot": 0.0,
        "floop_signal": 0,
        "floop_strength": 5.0,
        "floop_trend": 1,
        "floop_trend_aligned": 1,
    }
    sent_signals_path.write_text(json.dumps({
        "sig-1": {
            "time_sent": signal_time,
            "timeframe": "M15",
            "direction": "BULL",
            "type": "FVG",
            "ticket_id": ticket_id,
            "outcome_recorded": True,
            "features": features,
        }
    }))

    deal_in = SimpleNamespace(position_id=ticket_id, entry=0, price=2300.0, profit=0.0, commission=0.0, swap=0.0)
    deal_out = SimpleNamespace(position_id=ticket_id, entry=1, price=2310.0, profit=100.0, commission=0.0, swap=0.0)
    fake_mt5 = SimpleNamespace(
        DEAL_ENTRY_OUT=1,
        initialize=lambda: True,
        orders_get=lambda ticket=None: [],
        positions_get=lambda ticket=None: [],
        history_deals_get=lambda position=None: [deal_in, deal_out] if position == ticket_id else [],
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)

    fake_execution = SimpleNamespace(get_active_broker_symbol=lambda symbol: symbol)
    monkeypatch.setitem(sys.modules, "src.execution", fake_execution)

    with patch("src.telegram_bot.send_telegram_alert"), \
         patch("src.inference.check_and_trigger_retraining", return_value={"retrained": False}) as mock_retrain:
        result = process_mt5_history_feedback(
            sent_signals_file=str(sent_signals_path),
            labeled_data_path=str(labeled_data_path),
            return_details=True,
        )

    mock_retrain.assert_called_once_with(1)
    assert result["feedback_count"] == 1
    df = pd.read_csv(labeled_data_path)
    assert len(df) == 1
    assert df.iloc[0]["time"] == signal_time
    assert df.iloc[0]["label"] == 1
    assert df.iloc[0]["pnl_relative"] == 2.0
    registry = json.loads(sent_signals_path.read_text())
    assert registry["sig-1"]["status"] == "resolved"
    assert registry["sig-1"]["result"] == "tp"
    assert registry["sig-1"]["exit_category"] == "tp_profit"
    assert registry["sig-1"]["pnl_relative"] == 2.0
    assert registry["sig-1"]["net_profit"] == 100.0
    assert registry["sig-1"]["close_price"] == 2310.0


def test_process_mt5_history_feedback_backfills_recorded_dual_option_missing_from_csv(tmp_path, monkeypatch):
    sent_signals_path = tmp_path / "sent_signals.json"
    labeled_data_path = tmp_path / "labeled_setups.csv"
    signal_time = "2026-06-05 19:03:40"
    ticket_id = 123457
    features = {
        "timeframe": 60,
        "hour": 19,
        "day_of_week": 4,
        "setup_type": 0,
        "direction": 1,
        "entry_price": 4400.0,
        "sl_price": 4395.0,
        "tp_price": 4410.0,
        "risk_pips": 5.0,
        "atr_14": 8.0,
        "trend": 1,
        "relative_risk": 0.625,
        "killzone": 0,
        "fvg_width": 1.0,
        "relative_fvg_width": 0.125,
        "near_psychological_level": 1,
        "knn_prob_sig": 0.7,
        "knn_prob_opp": 0.3,
        "dist_entry_to_poc": 0.0,
        "dist_entry_to_nearest_poc": 0.0,
        "dist_entry_to_pp": 0.0,
        "dist_entry_to_nearest_pivot": 0.0,
        "floop_signal": 0,
        "floop_strength": 5.0,
        "floop_trend": 1,
        "floop_trend_aligned": 1,
    }
    sent_signals_path.write_text(json.dumps({
        "sig-1": {
            "time_sent": signal_time,
            "timeframe": "H1",
            "direction": "BULL",
            "type": "FVG",
            "ticket_a": ticket_id,
            "outcome_a_recorded": True,
            "outcome_recorded": True,
            "features_0.5": features,
        }
    }))

    deal_in = SimpleNamespace(position_id=ticket_id, entry=0, price=4400.0, profit=0.0, commission=0.0, swap=0.0)
    deal_out = SimpleNamespace(position_id=ticket_id, entry=1, price=4399.0, profit=-50.0, commission=0.0, swap=0.0)
    fake_mt5 = SimpleNamespace(
        DEAL_ENTRY_OUT=1,
        initialize=lambda: True,
        orders_get=lambda ticket=None: [],
        positions_get=lambda ticket=None: [],
        history_deals_get=lambda position=None: [deal_in, deal_out] if position == ticket_id else [],
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)

    fake_execution = SimpleNamespace(get_active_broker_symbol=lambda symbol: symbol)
    monkeypatch.setitem(sys.modules, "src.execution", fake_execution)

    with patch("src.telegram_bot.send_telegram_alert"), \
         patch("src.inference.check_and_trigger_retraining", return_value={"retrained": False}) as mock_retrain:
        result = process_mt5_history_feedback(
            sent_signals_file=str(sent_signals_path),
            labeled_data_path=str(labeled_data_path),
            return_details=True,
        )

    mock_retrain.assert_called_once_with(1)
    assert result["feedback_count"] == 1
    df = pd.read_csv(labeled_data_path)
    assert len(df) == 1
    assert df.iloc[0]["time"] == signal_time
    assert df.iloc[0]["label"] == 0
    assert df.iloc[0]["pnl_relative"] == -0.2
    registry = json.loads(sent_signals_path.read_text())
    assert registry["sig-1"]["status_a"] == "resolved"
    assert registry["sig-1"]["result_a"] == "cut_loss_early"
    assert registry["sig-1"]["exit_category_a"] == "cut_loss_early"
    assert registry["sig-1"]["pnl_relative_a"] == -0.2
    assert registry["sig-1"]["net_profit_a"] == -50.0
    assert registry["sig-1"]["close_price_a"] == 4399.0


def test_process_mt5_history_feedback_does_not_duplicate_recorded_dual_option_in_csv(tmp_path, monkeypatch):
    sent_signals_path = tmp_path / "sent_signals.json"
    labeled_data_path = tmp_path / "labeled_setups.csv"
    signal_time = "2026-06-09 11:35:45"
    ticket_id = 3245218014
    features = {
        "timeframe": 30,
        "hour": 4,
        "day_of_week": 1,
        "setup_type": 0,
        "direction": -1,
        "entry_price": 4328.3994999999995,
        "sl_price": 4334.756,
        "tp_price": 4324.043,
        "risk_pips": 6.3565000000007785,
        "atr_14": 9.248214285714441,
        "trend": 1,
        "relative_risk": 0.6873218768102677,
        "killzone": 3,
        "fvg_width": 2.048999999999978,
        "relative_fvg_width": 0.2215562849970976,
        "near_psychological_level": 0,
        "knn_prob_sig": 0.1660584067596697,
        "knn_prob_opp": 0.8339415932403305,
        "dist_entry_to_poc": -0.0019348697517533723,
        "dist_entry_to_nearest_poc": 0.001938620730364677,
        "dist_entry_to_pp": 0.0035524905960413818,
        "dist_entry_to_nearest_pivot": 0.003539915081005325,
        "floop_signal": 0,
        "floop_strength": 5.0,
        "floop_trend": 1,
        "floop_trend_aligned": 0,
    }
    existing_row = features.copy()
    existing_row.update({
        "time": signal_time,
        "entry_price": 4328.3995,
        "label": 0,
        "pnl_relative": -0.0880201368678093,
    })
    pd.DataFrame([existing_row]).to_csv(labeled_data_path, index=False)

    sent_signals_path.write_text(json.dumps({
        "sig-1": {
            "time_sent": signal_time,
            "timeframe": "M30",
            "direction": "BEAR",
            "type": "BPR",
            "ticket_a": ticket_id,
            "ticket_b": None,
            "outcome_a_recorded": True,
            "outcome_recorded": True,
            "features_0.5": features,
        }
    }))

    deal_in = SimpleNamespace(position_id=ticket_id, entry=0, price=features["entry_price"], profit=0.0, commission=0.0, swap=0.0)
    deal_out = SimpleNamespace(position_id=ticket_id, entry=1, price=4328.959, profit=-10199.0, commission=0.0, swap=0.0)
    fake_mt5 = SimpleNamespace(
        DEAL_ENTRY_OUT=1,
        initialize=lambda: True,
        orders_get=lambda ticket=None: [],
        positions_get=lambda ticket=None: [],
        history_deals_get=lambda position=None: [deal_in, deal_out] if position == ticket_id else [],
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)

    fake_execution = SimpleNamespace(get_active_broker_symbol=lambda symbol: symbol)
    monkeypatch.setitem(sys.modules, "src.execution", fake_execution)

    with patch("src.telegram_bot.send_telegram_alert") as mock_alert, \
         patch("src.inference.check_and_trigger_retraining", return_value={"retrained": False}) as mock_retrain:
        result = process_mt5_history_feedback(
            sent_signals_file=str(sent_signals_path),
            labeled_data_path=str(labeled_data_path),
            return_details=True,
        )

    mock_retrain.assert_not_called()
    mock_alert.assert_not_called()
    assert result["feedback_count"] == 0
    df = pd.read_csv(labeled_data_path)
    assert len(df) == 1


def test_analyze_trade_outcome_reason_marks_bep_plus_as_protected_not_tp():
    features = {
        "timeframe": 60,
        "hour": 16,
        "day_of_week": 0,
        "setup_type": 0,
        "direction": 1,
        "entry_price": 4337.100386,
        "sl_price": 4329.438,
        "tp_price": 4346.261,
        "risk_pips": 7.6623859999999695,
        "atr_14": 22.07271428571442,
        "trend": 1,
        "relative_risk": 0.3471428978247187,
        "killzone": 2,
        "fvg_width": 6.55199999999968,
        "relative_fvg_width": 0.2968370774517826,
        "near_psychological_level": 0,
        "knn_prob_sig": 0.4139727112406424,
        "knn_prob_opp": 0.5860272887593576,
        "dist_entry_to_poc": 0.00263999171228671,
        "dist_entry_to_nearest_poc": 0.002633040507169667,
        "dist_entry_to_pp": 0.0007872552912201377,
        "dist_entry_to_nearest_pivot": 0.0007866360078606851,
        "floop_signal": 0,
        "floop_strength": 11.0,
        "floop_trend": 1,
        "floop_trend_aligned": 1,
        "close_price": 4337.3000,
        "close_reason": "SL",
        "net_profit": 7250.02,
    }

    analysis = analyze_trade_outcome_reason(features, label=1, pnl_relative=0.026)

    assert "BEP+" in analysis
    assert "proteksi" in analysis.lower()
    assert "mencapai TP" not in analysis
    assert "Bobot Latih = 2.00" not in analysis


def test_process_mt5_history_feedback_reports_sl_protected_profit_not_tp_win(tmp_path, monkeypatch):
    sent_signals_path = tmp_path / "sent_signals.json"
    labeled_data_path = tmp_path / "labeled_setups.csv"
    signal_time = "2026-06-08 23:37:11"
    ticket_id = 3243059472
    features = {
        "timeframe": 60,
        "hour": 16,
        "day_of_week": 0,
        "setup_type": 0,
        "direction": 1,
        "entry_price": 4337.100386,
        "sl_price": 4329.438,
        "tp_price": 4346.261,
        "risk_pips": 7.6623859999999695,
        "atr_14": 22.07271428571442,
        "trend": 1,
        "relative_risk": 0.3471428978247187,
        "killzone": 2,
        "fvg_width": 6.55199999999968,
        "relative_fvg_width": 0.2968370774517826,
        "near_psychological_level": 0,
        "knn_prob_sig": 0.4139727112406424,
        "knn_prob_opp": 0.5860272887593576,
        "dist_entry_to_poc": 0.00263999171228671,
        "dist_entry_to_nearest_poc": 0.002633040507169667,
        "dist_entry_to_pp": 0.0007872552912201377,
        "dist_entry_to_nearest_pivot": 0.0007866360078606851,
        "floop_signal": 0,
        "floop_strength": 11.0,
        "floop_trend": 1,
        "floop_trend_aligned": 1,
    }
    close_price = features["entry_price"] + (features["entry_price"] - features["sl_price"]) * 0.026
    sent_signals_path.write_text(json.dumps({
        "sig-bep-plus": {
            "time_sent": signal_time,
            "timeframe": "H1",
            "direction": "BULL",
            "type": "BPR",
            "price_0.618": features["entry_price"],
            "ticket_b": ticket_id,
            "outcome_b_recorded": False,
            "outcome_recorded": False,
            "features_0.618": features,
        }
    }))

    deal_in = SimpleNamespace(position_id=ticket_id, entry=0, price=features["entry_price"], profit=0.0, commission=0.0, swap=0.0)
    deal_out = SimpleNamespace(position_id=ticket_id, entry=1, price=close_price, profit=7250.02, commission=0.0, swap=0.0, reason=4)
    fake_mt5 = SimpleNamespace(
        DEAL_ENTRY_OUT=1,
        DEAL_REASON_SL=4,
        DEAL_REASON_TP=5,
        initialize=lambda: True,
        orders_get=lambda ticket=None: [],
        positions_get=lambda ticket=None: [],
        history_deals_get=lambda position=None: [deal_in, deal_out] if position == ticket_id else [],
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)

    fake_execution = SimpleNamespace(get_active_broker_symbol=lambda symbol: symbol)
    monkeypatch.setitem(sys.modules, "src.execution", fake_execution)

    with patch("src.telegram_bot.send_telegram_alert") as mock_alert, \
         patch("src.inference.check_and_trigger_retraining", return_value={"retrained": False}):
        result = process_mt5_history_feedback(
            sent_signals_file=str(sent_signals_path),
            labeled_data_path=str(labeled_data_path),
            return_details=True,
        )

    assert result["feedback_count"] == 1
    df = pd.read_csv(labeled_data_path)
    assert df.iloc[0]["label"] == 1
    assert df.iloc[0]["pnl_relative"] == pytest.approx(0.026)

    message = mock_alert.call_args.args[0]
    assert "BEP+" in message
    assert "SL" in message
    assert "Trade berhasil mencapai TP" not in message
    assert "WIN (PROFIT)" not in message
