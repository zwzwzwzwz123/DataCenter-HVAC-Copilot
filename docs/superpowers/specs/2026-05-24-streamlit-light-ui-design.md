# Streamlit Light UI Design

## Goal

Refresh the Streamlit demo from a pure-black console into a minimal, premium light interface inspired by OpenAI and Apple: neutral, spacious, readable, and suitable for interview screenshots.

## Scope

Only the visual layer changes. The API client, Agent logic, evaluation logic, task options, and data flow remain unchanged.

## Design Direction

- Use a warm off-white app background instead of pure black.
- Use white cards with subtle borders for input, answer, status, expanders, metrics, and dataframes.
- Use near-black text for primary content and Apple-like gray for secondary text.
- Use a restrained accent color only for focus states, small status details, and selected interactions.
- Keep the existing layout: hero, boundary strip, two tabs, left Mission Control, right result area.
- Reduce the "hacker console" feel while preserving enough technical structure through small mono labels.

## Visual Tokens

- App background: `#f7f7f5`
- Surface: `#ffffff`
- Soft surface: `#f1f1ef`
- Hover: `#f5f5f2`
- Border: `#deded8`
- Soft border: `#ecece7`
- Primary text: `#1f1f1f`
- Muted text: `#6e6e73`
- Subtle text: `#9a9a92`
- Accent: `#10a37f`
- Warning: `#b7791f`

## Components

- Hero: white-space-led header with a small muted kicker, strong black title, and gray subtitle.
- Boundary strip: white card, subtle border, amber dot, compact text.
- Mission Control panel: white card, subtle shadow, gray form controls, black primary action.
- Answer panel: white content-first card with muted mono section label.
- Status cards: white compact cards with subtle hover border.
- Tabs: minimal underline style with selected tab using dark text and accent underline.
- Expanders and dataframes: white surfaces with light borders and minimal chrome.

## Acceptance Criteria

- The page no longer reads as pure black or dark console.
- Existing Streamlit controls and custom HTML still render correctly.
- Text remains readable on desktop and mobile.
- Existing tests still pass.
- The app remains accessible at `http://127.0.0.1:8501`.
