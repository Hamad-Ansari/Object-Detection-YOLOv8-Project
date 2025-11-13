# Contributing to Object Detection YOLOv8 Project

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs
1. Check if the bug has already been reported in [Issues](https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Error messages and logs

### Suggesting Features
1. Check existing feature requests
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach

### Pull Requests

#### Setup Development Environment
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Object-Detection-YOLOv8-Project.git
cd Object-Detection-YOLOv8-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create a new branch
git checkout -b feature/your-feature-name
```

#### Code Guidelines
1. **Python Style**
   - Follow PEP 8 style guide
   - Use meaningful variable names
   - Add docstrings to functions and classes
   - Keep functions focused and small

2. **Code Example**
```python
def detect_objects(image_path, conf_threshold=0.25):
    """
    Detect objects in an image.
    
    Args:
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        list: List of detected objects with bounding boxes
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    # Implementation
    pass
```

3. **Testing**
   - Add tests for new features
   - Ensure existing tests pass
   - Test on different platforms if possible

4. **Documentation**
   - Update README.md if needed
   - Add comments for complex logic
   - Update SETUP.md for new dependencies

#### Commit Messages
Use clear, descriptive commit messages:
```
Add feature: Real-time video detection with FPS counter

- Implemented FPS calculation
- Added display overlay
- Updated documentation
```

#### Submitting Pull Request
1. Push your changes to your fork
2. Create a pull request with:
   - Clear title and description
   - Reference related issues
   - List of changes made
   - Screenshots/videos if applicable

3. Wait for review and address feedback

### Code Review Process
- Maintainers will review your PR
- Address requested changes
- Once approved, your PR will be merged

## Development Areas

### High Priority
- [ ] Performance optimization
- [ ] Additional model architectures
- [ ] Mobile deployment support
- [ ] Real-time streaming improvements

### Medium Priority
- [ ] Web interface
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Additional dataset converters

### Low Priority
- [ ] Additional visualization options
- [ ] Model compression techniques
- [ ] Benchmark comparisons
- [ ] Tutorial videos

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Contact: mrhammadzahid24@gmail.com

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🎉
