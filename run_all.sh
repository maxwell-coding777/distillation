#!/usr/bin/env bash
# Idempotent pipeline driver: safe to re-run after a container restart, resumes wherever it stopped.
set -u
cd "$(dirname "$0")"; R=runs; mkdir -p $R results
EPOCHS=${EPOCHS:-40}
running() { pgrep -f "$1" >/dev/null; }
if [ ! -f $R/teacher/weights/best.pt ] || ! grep -q "^ *8," $R/teacher/results.csv 2>/dev/null; then
  running train_teacher.py || (cd $R && nohup python3 ../src/train_teacher.py >> teacher.log 2>&1 &)
  echo "teacher training"; exit 0
fi
running train_teacher.py && { echo "teacher finishing"; exit 0; }
[ -f $R/teacher_eval.json ] || (cd src && python3 eval_teacher.py > ../$R/teacher_eval.log 2>&1)
# KD run is ~4x more expensive per step (teacher forward): give it 3 threads, baseline 1 -> similar wall time
done_run() { python3 -c "import json,sys; h=json.load(open('$R/$1/history.json')); sys.exit(0 if h[-1]['epoch']>=$EPOCHS else 1)" 2>/dev/null; }
done_run student_none || running "kd none" || (cd src && nohup python3 train_student.py --kd none --epochs $EPOCHS --threads 1 --resume >> ../$R/student_none.log 2>&1 &)
done_run student_full || running "kd full" || (cd src && nohup python3 train_student.py --kd full --epochs $EPOCHS --threads 3 --resume >> ../$R/student_full.log 2>&1 &)
if done_run student_none && done_run student_full; then echo "students done"; else echo "students training"; fi
