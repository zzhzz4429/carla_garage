#!/bin/bash

# Enhanced Experiment Setup Script
# Usage: ./setup_experiment_with_scenario.sh <condition> <scenario> <participant_id> <trial_number>
#
# Conditions: fusion, external_only, internal_only, baseline
# Scenarios: SC-EV, SC-INT, SC-PED
# Example: ./setup_experiment_with_scenario.sh fusion SC-EV P001 1

if [ $# -ne 4 ]; then
    echo "Usage: $0 <condition> <scenario> <participant_id> <trial_number>"
    echo ""
    echo "Conditions:"
    echo "  fusion        - Full adaptive risk fusion"
    echo "  external_only - TTC + signal compliance only"
    echo "  internal_only - Driver state monitoring only"
    echo "  baseline      - No alerts (control condition)"
    echo ""
    echo "Scenarios:"
    echo "  SC-EV   - Emergency vehicle (parked police cruiser)"
    echo "  SC-INT  - Intersection (red-to-green signal)"
    echo "  SC-PED  - Pedestrian (mid-block crossing)"
    echo ""
    echo "Example: $0 fusion SC-EV P001 1"
    exit 1
fi

CONDITION=$1
SCENARIO=$2
PARTICIPANT_ID=$3
TRIAL_NUMBER=$4

# Validate condition
case $CONDITION in
    fusion|external_only|internal_only|baseline)
        echo "✅ Valid condition: $CONDITION"
        ;;
    *)
        echo "❌ Invalid condition: $CONDITION"
        echo "   Must be: fusion, external_only, internal_only, or baseline"
        exit 1
        ;;
esac

# Validate scenario
case $SCENARIO in
    SC-EV|SC-INT|SC-PED)
        echo "✅ Valid scenario: $SCENARIO"
        ;;
    *)
        echo "❌ Invalid scenario: $SCENARIO"
        echo "   Must be: SC-EV, SC-INT, or SC-PED"
        exit 1
        ;;
esac

# Set environment variables
export ENABLE_TTC_LOGGING=true
export EXPERIMENT_CONDITION=$CONDITION
export SCENARIO_NAME=$SCENARIO
export PARTICIPANT_ID=$PARTICIPANT_ID
export TRIAL_NUMBER=$TRIAL_NUMBER

# Set external risk disable flag for internal_only condition
if [ "$CONDITION" = "internal_only" ]; then
    export DISABLE_EXTERNAL_RISK=true
    echo "🔇 External risk disabled for internal_only condition"
else
    export DISABLE_EXTERNAL_RISK=false
    echo "✅ External risk enabled"
fi

# Set alert disable flag for baseline condition
if [ "$CONDITION" = "baseline" ]; then
    export DISABLE_ALL_ALERTS=true
    echo "🔇 All alerts disabled for baseline condition"
else
    export DISABLE_ALL_ALERTS=false
    echo "✅ Alerts enabled"
fi

echo ""
echo "🧪 EXPERIMENT CONFIGURATION"
echo "================================="
echo "Condition:           $CONDITION"
echo "Scenario:            $SCENARIO"
echo "Participant ID:      $PARTICIPANT_ID"
echo "Trial Number:        $TRIAL_NUMBER"
echo "TTC Logging:         $ENABLE_TTC_LOGGING"
echo "External Risk:       $([ "$DISABLE_EXTERNAL_RISK" = "true" ] && echo "DISABLED" || echo "ENABLED")"
echo "Alerts:              $([ "$DISABLE_ALL_ALERTS" = "true" ] && echo "DISABLED" || echo "ENABLED")"
echo ""
echo "Log filename will be: ttc_${SCENARIO}_${CONDITION}_${PARTICIPANT_ID}_T${TRIAL_NUMBER}_[timestamp].json"
echo "Log directory: /home/ascc304/carla_garage/experiment_logs/zhaohua"
echo ""
echo "🚀 Environment configured! Ready to run experiment."
echo "   Start CARLA with: python sensor_agent.py"
echo ""

# Scenario-specific instructions
case $SCENARIO in
    SC-EV)
        echo "📋 SC-EV Scenario Instructions:"
        echo "   - Emergency vehicle parked on shoulder"
        echo "   - Test lane-change decision timing"
        echo "   - Driver states: attentive vs sleepy"
        ;;
    SC-INT)
        echo "📋 SC-INT Scenario Instructions:"
        echo "   - Four-way signalized intersection"
        echo "   - Test signal compliance behavior"
        echo "   - Driver distraction: phone use starting 5s before signal"
        ;;
    SC-PED)
        echo "📋 SC-PED Scenario Instructions:"
        echo "   - Pedestrian mid-block crossing"
        echo "   - Test collision avoidance timing"
        echo "   - Driver distraction: reaching back starting 5s before spawn"
        ;;
esac

echo "" 