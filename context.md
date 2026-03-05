# An app that allows user to learn what is their best way of taking photos

## Problem Statement
Many people don't actually know how to take good photos of themselves. Most of the time, they're clueless and embarrassed to take multiple angles of photos just to not be satisfied with all of them, or worst of all, blindly accepting a bad photo of themselves. 

## The Goal
We want to create an AI app that easily and realistically shows them how they could take better photos of themselves by using a simulated environßment with an interface akin to blender that gives them many tools like editing the photo and using presets to show a realistic photo of themselves in great poses. 

## Challenges 
Using LLMs may be wildly inaccurate and would gave an eerie vibe of generating a icky AI photo of themselves. As a result, we may be trying to develop technology that is really tricky to pull off. A reasonable idea we came up is to use multiple photos of different angles that allows the AI to generate an accurate photo of themselves. 

## Detailed Plan 

Plan: AI Photo Pose Coach — Hackathon MVP
A web app where users upload 1–3 selfies, pick from preset "good poses" (e.g., 3/4 angle, profile, headshot), and get a realistic re-posed photo of themselves generated via InstantID + ControlNet-OpenPose on SDXL. A FastAPI backend handles ML inference; a React frontend provides pose selection, lighting controls, and before/after comparison. Scoped to be demoable in 2 weeks.

Key AI Decision: Image Generation > 3D Reconstruction
3D approaches (NeRF, Gaussian Splatting) need 50–200 input photos and heavy compute — infeasible for a hackathon. Instead, use identity-preserving diffusion: the user uploads 1–3 photos, and InstantID generates new photos of them in different poses that still look like real photographs. This gives the "it's clearly me" factor without the uncanny 3D avatar look.

Steps
Set up the ML backend (FastAPI + Python) — Integrate InstantID (SDXL-based, zero-shot identity preservation from a single face image) with ControlNet-OpenPose for pose control and DWPose for extracting pose skeletons from uploaded photos. Host on a single T4/A10 GPU via RunPod, Modal, or Replicate.

Build a preset pose library — Curate 6–8 reference pose skeletons (3/4 angle, profile, looking up, professional headshot, casual outdoor, dating-app smile, etc.) stored as OpenPose keypoint JSON files. These become the ControlNet conditioning inputs.

Build the React frontend — Photo upload UI, a pose-preset picker (thumbnail grid), lighting dropdown (warm sunlight / studio / golden hour — passed as prompt text), and a before/after comparison view. Use React + Tailwind + shadcn/ui for speed.

Wire frontend → backend API — POST /generate accepts the uploaded face image, selected pose preset, and lighting choice; returns the generated image. Add loading states (expect ~15–30s per generation).

Add educational "what changed" annotations — Overlay simple text/arrows on the result explaining why this angle works (e.g., "3/4 angle slims the face," "chin slightly down avoids double chin"). This is the learning component that differentiates your app.

Polish & stretch goals — If ahead of schedule: add an interactive 2D pose editor with react-konva (drag skeleton joints), add IC-Light relighting as a second pass, or swap SDXL for FLUX + PuLID for higher quality output.

Further Considerations
Compute hosting for demo day — Replicate has pre-hosted InstantID as an API (easiest, ~$0.01/run) vs. self-hosting on RunPod (more control, ~$0.50/hr for a T4). Which fits your budget better?
Generation speed vs. quality — SDXL + InstantID gives ~15–30s per image on a T4. Acceptable for a demo? If judges expect near-instant results, you could pre-generate a few examples and cache them.
Scope of "learning" — Should the app just show better photos, or also explain photography principles (rule of thirds, jawline angles, lighting direction)? Adding a tips/explanation panel alongside results would strengthen the educational angle and differentiate from a generic AI photo tool.
## 

