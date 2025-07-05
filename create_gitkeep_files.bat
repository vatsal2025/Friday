@echo off
echo Creating .gitkeep files in storage directories...

type nul > storage\memory\.gitkeep
type nul > storage\models\.gitkeep
type nul > storage\data\.gitkeep
type nul > storage\logs\.gitkeep

echo .gitkeep files created successfully.
pause