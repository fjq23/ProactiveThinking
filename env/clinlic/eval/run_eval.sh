# Set environment variables
export DOCTOR_MODEL=Qwen3.6-35B-A3B
export DOCTOR_TYPE=proactive
export RUN_NUMBER=4
export MAX_SAMPLE=1000
export MAX_WORKERS=64
export SPEC_COT=false # 预测患者回复时是否使用COT
# export SPEC_BRANCH_NUM=3 # 实验模式: 预测并执行 1-5 个患者回复分支; 不设置则使用默认逻辑
export INPUT_FILE=/root/wat/AgentClinic/agentclinic_medqa_extended.jsonl
BASENAME="$(basename "$INPUT_FILE")"
INPUT_FILE_NAME="${BASENAME%.*}"
# export GAMMA=1.5
# export OUTPUT_FILE=${DOCTOR_MODEL}_${INPUT_FILE_NAME}_${DOCTOR_TYPE}_${GAMMA}.jsonl

## Run test
# bash test_inquiry.sh

for x in 1.0 1.5 2.0 3.0 5.0
do
export GAMMA=${x} #投机解码宽容度
BRANCH_SUFFIX=""
if [ -n "${SPEC_BRANCH_NUM:-}" ]; then
BRANCH_SUFFIX="_branch${SPEC_BRANCH_NUM}"
fi
export OUTPUT_FILE=${DOCTOR_MODEL}_${INPUT_FILE_NAME}_${DOCTOR_TYPE}_${GAMMA}_${SPEC_COT}${BRANCH_SUFFIX}.jsonl

# Run test
bash test_inquiry.sh
done
