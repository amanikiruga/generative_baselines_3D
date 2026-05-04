#!/usr/bin/env bash
# Print a per-dataset checklist of expected artifacts.
# Usage: ./status.sh           (one-shot)
#        ./status.sh --watch   (refresh every 60s + collect_results + build_index)

set -uo pipefail

OUT=${OUT:-/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/eval_outputs}
WMNEW=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new

DATASETS=(re10k dl3dv dl3dv_test tanksandtemples scannetpp aria vkitti2 scenenet_depth spatialvid_nvs agibot_world)

# task gates (mirrors run_dataset.sh)
has_depth() { case $1 in dl3dv|aria|vkitti2|scannetpp|scenenet_depth) return 0;; *) return 1;; esac; }
has_nvs()   { case $1 in re10k|dl3dv|dl3dv_test|tanksandtemples|aria|vkitti2|scannetpp|spatialvid_nvs|agibot_world) return 0;; *) return 1;; esac; }

# baseline gates
b_has() {
  local ds=$1 b=$2
  case "$b:$ds" in
    geo4d_pose:scenenet_depth)  return 1;;
    geo4d_pose:*)               return 0;;
    raydiff:scenenet_depth)     return 1;;
    raydiff:*)                  return 0;;
    geo4d_depth:spatialvid_nvs|geo4d_depth:agibot_world) return 1;;
    chrono:spatialvid_nvs|chrono:agibot_world)           return 1;;
    geo4d_depth:re10k|geo4d_depth:dl3dv_test|geo4d_depth:tanksandtemples) return 1;;
    geo4d_depth:*)              return 0;;
    chrono:re10k|chrono:dl3dv_test|chrono:tanksandtemples) return 1;;
    chrono:*)                   return 0;;
    gen3c:dl3dv|gen3c:re10k|gen3c:dl3dv_test|gen3c:tanksandtemples|gen3c:aria|gen3c:scannetpp|gen3c:spatialvid_nvs|gen3c:agibot_world) return 0;;
    gen3c:*)                    return 1;;
    dfot:re10k|dfot:aria)       return 0;;
    dfot:*)                     return 1;;
    seva:dl3dv|seva:re10k|seva:dl3dv_test|seva:tanksandtemples|seva:aria|seva:scannetpp|seva:spatialvid_nvs|seva:agibot_world) return 0;;
    seva:*)                     return 1;;
    wan:dl3dv|wan:re10k|wan:dl3dv_test|wan:tanksandtemples|wan:aria|wan:scannetpp|wan:spatialvid_nvs|wan:agibot_world) return 0;;
    wan:*)                      return 1;;
  esac
}

# mark "done" iff dir exists, has ≥1 sample_ subdir AND a summary file
mark() {
  local d=$1
  if [ ! -d "$d" ]; then echo "[ ]"; return; fi
  local n=$(find "$d" -maxdepth 1 -type d -name 'sample_*' 2>/dev/null | wc -l)
  if [ "$n" -ge 1 ]; then echo "[x]"; else echo "[~]"; fi
}

print_status() {
  printf "%s  status — %s\n\n" "$(date '+%F %T')" "$OUT"
  for ds in "${DATASETS[@]}"; do
    echo "## $ds"
    for METHOD in ours ours_nvs_only; do
      printf "  %-14s pose_depth %s" "$METHOD" "$(mark $OUT/$ds/${METHOD}_pose_depth)"
      printf "  pose_eval %s"             "$(mark $OUT/$ds/${METHOD}_pose_eval)"
      if has_depth $ds; then printf "  depth_eval %s" "$(mark $OUT/$ds/${METHOD}_depth_eval)"; fi
      if has_nvs   $ds; then
        printf "  nvs %s"     "$(mark $OUT/$ds/${METHOD}_nvs)"
        printf "  nvs_eval %s" "$(mark $OUT/$ds/${METHOD}_nvs_eval)"
      fi
      echo
    done
    printf "  baselines     "
    for b in geo4d_pose raydiff geo4d_depth chrono gen3c dfot seva wan; do
      if b_has $ds $b; then
        case $b in
          geo4d_pose)  dir=geo4d_pose ;;
          raydiff)     dir=raydiffusion_pose ;;
          geo4d_depth) dir=geo4d_depth ;;
          chrono)      dir=chronodepth_depth ;;
          gen3c)       dir=gen3c_nvs ;;
          dfot)        dir=dfot_nvs ;;
          seva)        dir=seva_nvs ;;
          wan)         dir=wan_flf_nvs ;;
        esac
        printf "%s%s " "$b" "$(mark $OUT/$ds/$dir)"
      fi
    done
    echo; echo
  done
  echo "Legend: [x]=done, [~]=dir but no samples yet, [ ]=not started"
}

if [ "${1:-}" = "--watch" ]; then
  while true; do
    clear
    print_status
    echo
    echo "Refreshing aggregate (collect_results.py + build_index.py)..."
    (cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines && \
       mamba run -n test2 python collect_results.py >/dev/null 2>&1 && \
       mamba run -n test2 python build_index.py     >/dev/null 2>&1) \
      && echo "  aggregate refreshed." || echo "  aggregate refresh FAILED (see logs)."
    sleep 60
  done
else
  print_status
fi
