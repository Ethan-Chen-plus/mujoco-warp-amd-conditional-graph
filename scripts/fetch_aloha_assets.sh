#!/usr/bin/env bash
set -euo pipefail

# Fetch the ALOHA benchmark assets from the public MuJoCo Menagerie source.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST="${DEST:-$ROOT/upstream/mujoco_warp/benchmarks/aloha/assets}"
BASE_URL="${BASE_URL:-https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/aloha/assets}"

FILES=(
  angled_extrusion.stl
  corner_bracket.stl
  d405_solid.stl
  extrusion_1000.stl
  extrusion_1220.stl
  extrusion_150.stl
  extrusion_2040_1000.stl
  extrusion_2040_880.stl
  extrusion_600.stl
  interbotix_black.png
  overhead_mount.stl
  small_meta_table_diffuse.png
  tablelegs.obj
  tabletop.obj
  vx300s_1_base.stl
  vx300s_2_shoulder.stl
  vx300s_3_upper_arm.stl
  vx300s_4_upper_forearm.stl
  vx300s_5_lower_forearm.stl
  vx300s_6_wrist.stl
  vx300s_7_gripper.stl
  vx300s_7_gripper_bar.stl
  vx300s_7_gripper_camera.stl
  vx300s_7_gripper_prop.stl
  vx300s_7_gripper_prop_bar.stl
  vx300s_7_gripper_wrist_mount.stl
  vx300s_8_custom_finger_left.stl
  vx300s_8_custom_finger_right.stl
  wormseye_mount.stl
)

mkdir -p "$DEST"
for file in "${FILES[@]}"; do
  curl --fail --location --retry 3 --retry-delay 2 \
    --output "$DEST/$file" "$BASE_URL/$file"
done

sha256sum "${FILES[@]/#/$DEST/}" > "$DEST/SHA256SUMS"
printf 'Fetched %d ALOHA assets into %s\n' "${#FILES[@]}" "$DEST"
