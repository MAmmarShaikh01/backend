import os
import asyncio
import sys
from groq import AsyncGroq
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Documentation structure for Physical AI & Humanoid Robotics
DOCS_STRUCTURE = {
    "module-1": [
        {
            "filename": "classification-models.md",
            "title": "Classification Models for Robotics",
            "sidebar_position": 3,
            "topic": "Build classification models for robot behavior recognition, object identification, and state prediction. Cover decision trees, random forests, SVM, and gradient boosting with practical robotics examples.",
            "key_points": [
                "Binary and multi-class classification in robotics",
                "Decision trees and random forests for robot behavior classification",
                "Support Vector Machines for object recognition",
                "Gradient boosting (XGBoost, LightGBM) for high accuracy",
                "Model evaluation metrics (accuracy, precision, recall, F1)",
                "Handling imbalanced datasets in robotics",
                "Feature importance and interpretability",
                "Practical project: Robot behavior classifier with 95%+ accuracy"
            ]
        },
        {
            "filename": "regression-models.md",
            "title": "Regression Models for Robot Performance",
            "sidebar_position": 4,
            "topic": "Learn regression techniques to predict robot performance, movement efficiency, and system response times. Cover linear regression, polynomial regression, ensemble methods with real robotics datasets.",
            "key_points": [
                "Linear and polynomial regression fundamentals",
                "Multiple regression with robotics features",
                "Regularization (Lasso, Ridge) to prevent overfitting",
                "Ensemble regression methods (Random Forest, Gradient Boosting)",
                "Time-to-completion prediction models",
                "System performance impact on robot behavior",
                "Model evaluation (RMSE, MAE, R² score)",
                "Practical project: Robot movement efficiency prediction system"
            ]
        },
        {
            "filename": "time-series.md",
            "title": "Time-Series Analysis for Robot Monitoring",
            "sidebar_position": 5,
            "topic": "Master time-series forecasting for robot movement patterns, sensor data trends, and system monitoring. Cover ARIMA, Prophet, LSTM networks for sequential robotics data.",
            "key_points": [
                "Time-series data in robotics (movement, sensors, performance)",
                "Stationarity and differencing",
                "ARIMA models for robot behavior forecasting",
                "Facebook Prophet for seasonal patterns",
                "LSTM neural networks for complex sequences",
                "Multi-variate time-series analysis",
                "Forecasting sensor data and system needs",
                "Practical project: 30-day robot performance prediction system"
            ]
        },
        {
            "filename": "ml-project.md",
            "title": "Mini-Project: Robot Behavior Prediction System",
            "sidebar_position": 6,
            "topic": "Build a complete end-to-end machine learning system for early robot behavior prediction using sensor data and environmental factors. Integrate data collection, preprocessing, model training, and deployment.",
            "key_points": [
                "Project overview: Early robot behavior prediction system",
                "Dataset collection and preparation",
                "Feature engineering from sensor data",
                "Multi-model comparison and selection",
                "Hyperparameter tuning with GridSearch/RandomSearch",
                "Model deployment strategies",
                "Building a simple prediction API with FastAPI",
                "Real-time monitoring dashboard"
            ]
        }
    ],
    "module-2": [
        {
            "filename": "cv-intro.md",
            "title": "Introduction to Computer Vision in Robotics",
            "sidebar_position": 1,
            "topic": "Explore how computer vision revolutionizes robotics through automated object detection, environment mapping, and visual servoing. Learn image fundamentals and robotic vision systems.",
            "key_points": [
                "Computer vision applications in robotics",
                "Image formation and digital representation",
                "Color spaces (RGB, HSV, LAB) for robotic vision",
                "Image acquisition systems (cameras, stereo, depth sensors)",
                "Lighting conditions and image quality",
                "Robotic vision best practices",
                "Common robotic image datasets (COCO, KITTI, etc.)",
                "Introduction to OpenCV for robotic vision processing"
            ]
        },
        {
            "filename": "image-processing.md",
            "title": "Image Acquisition and Preprocessing for Robotics",
            "sidebar_position": 2,
            "topic": "Learn essential image preprocessing techniques for robotic vision including filtering, segmentation, feature extraction, and background removal for robust object recognition.",
            "key_points": [
                "Image enhancement and noise reduction",
                "Background removal for object isolation",
                "Color-based segmentation for object detection",
                "Morphological operations (erosion, dilation)",
                "Edge detection for object boundaries",
                "Contour detection and analysis",
                "Feature extraction (color histograms, texture, shape)",
                "Practical project: Automated object segmentation"
            ]
        },
        {
            "filename": "deep-learning-cnn.md",
            "title": "Deep Learning for Object Recognition in Robotics",
            "sidebar_position": 3,
            "topic": "Master Convolutional Neural Networks (CNNs) for object recognition in robotics. Learn transfer learning with pre-trained models, data augmentation, and deployment strategies.",
            "key_points": [
                "CNN architecture fundamentals (convolution, pooling, FC layers)",
                "Building CNNs with TensorFlow/Keras and PyTorch",
                "Transfer learning with ResNet, VGG, EfficientNet",
                "Data augmentation for limited robotic datasets",
                "Training strategies and regularization",
                "Multi-class object classification",
                "Model interpretation with Grad-CAM",
                "Practical project: 20+ object classifier with 98%+ accuracy"
            ]
        },
        {
            "filename": "object-detection.md",
            "title": "Object Detection for Robotic Navigation",
            "sidebar_position": 4,
            "topic": "Implement object detection models for obstacle detection, landmark recognition, and robotic navigation. Cover YOLO, Faster R-CNN, and custom detection pipelines.",
            "key_points": [
                "Object detection vs classification vs segmentation",
                "YOLO architecture and real-time detection",
                "Faster R-CNN for precise localization",
                "Training custom object detectors",
                "Annotation tools (LabelImg, CVAT) for robotic data",
                "Obstacle detection and avoidance",
                "Landmark recognition for navigation",
                "Practical project: Automated obstacle detection and classification"
            ]
        },
        {
            "filename": "cv-project.md",
            "title": "Mini-Project: Automated Robot Vision System",
            "sidebar_position": 5,
            "topic": "Build a complete automated vision system that measures object dimensions, performs scene analysis, color analysis, and tracking from image sequences using computer vision.",
            "key_points": [
                "Project overview: High-throughput vision system",
                "Multi-view image capture setup",
                "Object segmentation and 3D reconstruction",
                "Automated measurement extraction (dimensions, position, object count)",
                "Object analysis for environment assessment",
                "Color analysis for identification",
                "Time-lapse tracking",
                "Export data for ML analysis"
            ]
        }
    ],
    "module-3": [
        {
            "filename": "genomics-intro.md",
            "title": "Introduction to AI in Robot Control Systems",
            "sidebar_position": 1,
            "topic": "Understand how AI accelerates robot control systems, from sensor fusion to behavior prediction. Learn control system fundamentals and AI integration concepts.",
            "key_points": [
                "Robot control systems fundamentals (sensors, actuators, feedback)",
                "Control data formats and protocols (ROS messages, etc.)",
                "High-throughput sensor technologies",
                "AI applications in control (path planning, feedback control, prediction)",
                "Sensor to behavior mapping",
                "System identification and modeling",
                "Introduction to control frameworks and simulation environments",
                "Practical example: Analyzing robot sensor data"
            ]
        },
        {
            "filename": "sequence-analysis.md",
            "title": "Deep Learning for Robot Movement Sequences",
            "sidebar_position": 2,
            "topic": "Apply deep learning to robot movement sequence analysis including trajectory prediction, movement optimization, and behavior modeling using CNNs and RNNs on movement data.",
            "key_points": [
                "Encoding movement sequences for neural networks (poses, joint angles, trajectories)",
                "CNNs for pattern discovery in movement sequences",
                "RNNs and LSTMs for movement modeling",
                "Transformer models for long-range dependencies",
                "Trajectory prediction and optimization",
                "Movement pattern detection with deep learning",
                "Behavior prediction from movement sequences",
                "Practical project: Movement prediction from trajectory data"
            ]
        },
        {
            "filename": "crispr-ai.md",
            "title": "AI for Robot Hardware Optimization",
            "sidebar_position": 3,
            "topic": "Learn how AI optimizes robot hardware configurations by predicting component efficiency, off-target effects, and designing optimal hardware strategies for task performance.",
            "key_points": [
                "Robot hardware fundamentals and component integration",
                "Hardware design challenges",
                "ML models for component performance prediction",
                "Off-target effect prediction and minimization",
                "Deep learning for hardware efficiency scoring",
                "Multiplexed hardware strategy optimization",
                "AI-designed robots: examples and case studies",
                "Practical project: Design optimal hardware configurations for task efficiency"
            ]
        },
        {
            "filename": "genomic-selection.md",
            "title": "Robot Design and Configuration Selection",
            "sidebar_position": 4,
            "topic": "Master robot design selection techniques to accelerate robot development. Use ML to predict performance values and design optimal configurations for desired tasks.",
            "key_points": [
                "Traditional design vs AI-assisted selection",
                "Performance prediction models (regression, ensemble methods)",
                "Deep learning for complex performance prediction",
                "Multi-task and multi-environment models",
                "Training data design and optimization",
                "Performance prediction accuracy",
                "Optimal configuration selection with genetic algorithms",
                "Practical project: Predict robot performance from configuration parameters"
            ]
        },
        {
            "filename": "genomics-project.md",
            "title": "Mini-Project: Robot Performance Prediction System",
            "sidebar_position": 5,
            "topic": "Build a robot performance prediction system that predicts robot behaviors from configuration data using ensemble ML methods and validates predictions with real datasets.",
            "key_points": [
                "Project overview: Configuration-to-performance prediction pipeline",
                "Dataset: Robot configuration and performance data",
                "Parameter filtering and quality control",
                "Feature engineering from robot configurations",
                "Ensemble methods for performance prediction",
                "Cross-validation with different environments",
                "Model interpretation: identifying key parameters",
                "Integration with robot design workflows"
            ]
        }
    ],
    "module-4": [
        {
            "filename": "iot-intro.md",
            "title": "Introduction to IoT in Robotics",
            "sidebar_position": 1,
            "topic": "Learn IoT fundamentals for smart robotics including sensor networks, edge computing, and cloud integration for real-time robot monitoring and decision-making.",
            "key_points": [
                "IoT architecture for robotics",
                "Sensor types (position, IMU, camera, LiDAR)",
                "Microcontrollers (Arduino, Raspberry Pi, ESP32)",
                "Wireless communication (WiFi, Bluetooth, Zigbee)",
                "Edge computing vs cloud processing",
                "Data protocols (MQTT, ROS2, HTTP)",
                "Power management and battery solutions",
                "Practical example: Build a robot sensor monitoring node"
            ]
        },
        {
            "filename": "sensor-networks.md",
            "title": "Robot Sensor Networks and Data Collection",
            "sidebar_position": 2,
            "topic": "Design and deploy multi-sensor networks for comprehensive robot monitoring. Learn data aggregation, storage, and real-time streaming to ML pipelines.",
            "key_points": [
                "Sensor network topology design",
                "Multi-sensor data fusion",
                "Time-series database (InfluxDB, TimescaleDB)",
                "Real-time data streaming with Apache Kafka",
                "Data validation and quality control",
                "Missing data handling in sensor networks",
                "Network reliability and fault tolerance",
                "Practical project: 10-node robot sensor network"
            ]
        },
        {
            "filename": "yield-prediction.md",
            "title": "AI-Powered Robot Performance Prediction",
            "sidebar_position": 3,
            "topic": "Build accurate robot performance prediction models combining sensor data, environmental factors, and historical records using ensemble ML and deep learning.",
            "key_points": [
                "Multi-source data integration (sensors, environment, tasks)",
                "Feature engineering from sensor data (position, orientation, etc.)",
                "Environmental data APIs and forecasting",
                "Ensemble models combining multiple data sources",
                "Deep learning on multi-modal data",
                "Spatial and temporal modeling",
                "Pre-task performance estimation for planning",
                "Practical project: Robot performance forecasting"
            ]
        },
        {
            "filename": "smart-irrigation.md",
            "title": "Automated Robot Resource Management",
            "sidebar_position": 4,
            "topic": "Develop AI-driven resource management systems that optimize power usage using battery levels, task requirements, and environmental conditions with reinforcement learning.",
            "key_points": [
                "Power consumption and robot energy needs",
                "Battery sensor placement and interpretation",
                "Environmental condition integration",
                "Rule-based vs ML-based power management",
                "Reinforcement learning for energy scheduling",
                "Multi-resource: combined power and task optimization",
                "Hardware: power management, controllers, sensors",
                "Practical project: Smart power management system with 30% energy savings"
            ]
        },
        {
            "filename": "capstone-project.md",
            "title": "Capstone Project: Complete Robot Control System",
            "sidebar_position": 5,
            "topic": "Integrate everything learned into a comprehensive robot control platform with multi-sensor monitoring, behavior prediction, performance optimization, and automated control with web dashboard.",
            "key_points": [
                "System architecture: sensors, edge devices, cloud, web app",
                "Real-time monitoring dashboard (React + Chart.js)",
                "Behavior prediction from sensor feeds",
                "Automated alerts and notifications",
                "Historical data analysis and trends",
                "Performance prediction and task planning",
                "Automated control system integration",
                "Deployment: Docker, cloud hosting, mobile access",
                "Future enhancements: scaling to commercial robots"
            ]
        }
    ]
}

async def generate_lesson_content(client, module_name, lesson_info):
    """Generate comprehensive lesson content using Groq AI."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = f"""Create a comprehensive, educational markdown lesson for a university-level textbook on "Physical AI & Humanoid Robotics".

Module: {module_name}
Title: {lesson_info['title']}
Topic: {lesson_info['topic']}

Key points to cover:
{chr(10).join(f"- {point}" for point in lesson_info['key_points'])}

Requirements:
1. Start with frontmatter:
---
sidebar_position: {lesson_info['sidebar_position']}
---

2. Include these sections:
   - Introduction with real-world motivation
   - Core concepts with clear explanations
   - Multiple code examples using Python (scikit-learn, TensorFlow/PyTorch, pandas, numpy)
   - Practical applications in robotics/AI
   - Best practices and common pitfalls
   - Hands-on example or mini-project
   - Summary table or checklist
   - Next steps and further reading

3. Writing style:
   - Clear, engaging, educational tone
   - Practical examples from robotics/AI
   - Code examples that actually work and are well-commented
   - Use tables, lists, and formatting for readability
   - Include specific robot examples (humanoid robots, manipulators, mobile robots, etc.)
   - Add emojis sparingly (🤖 💡 ⚠️) only where appropriate

4. Code quality:
   - All code must be runnable and practical
   - Include imports and setup
   - Add comments explaining key steps
   - Show expected outputs

5. Length: Comprehensive (2000-3000 words minimum)

Generate the complete lesson content now:"""

            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "system",
                    "content": "You are an expert educator in AI, Physical AI, and humanoid robotics. "
                              "Create comprehensive, practical, code-rich lessons that teach both theory and implementation."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.4,
                max_tokens=8000
            )

            await asyncio.sleep(2)  # Rate limiting
            return response.choices[0].message.content

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      Retry {attempt + 1}/{max_retries} after error: {str(e)[:100]}")
                await asyncio.sleep(5 * (attempt + 1))
            else:
                print(f"      ✗ Error after {max_retries} attempts: {e}")
                return None

async def generate_all_lessons():
    """Generate all missing lesson content."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    docs_dir = Path("../website/docs")

    print("🤖 Starting AI-powered lesson generation for Physical AI & Humanoid Robotics...")
    print(f"📁 Target directory: {docs_dir.absolute()}\n")

    total_lessons = sum(len(lessons) for lessons in DOCS_STRUCTURE.values())
    current = 0

    for module_name, lessons in DOCS_STRUCTURE.items():
        module_dir = docs_dir / module_name
        module_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📚 Module: {module_name}")
        print(f"   Creating {len(lessons)} lessons...\n")

        for lesson_info in lessons:
            current += 1
            lesson_path = module_dir / lesson_info['filename']

            # Skip if already exists
            if lesson_path.exists():
                print(f"   [{current}/{total_lessons}] ⏭️  Skipping {lesson_info['filename']} (already exists)")
                continue

            print(f"   [{current}/{total_lessons}] 🤖 Generating {lesson_info['filename']}...")
            print(f"      Title: {lesson_info['title']}")

            content = await generate_lesson_content(client, module_name, lesson_info)

            if content:
                lesson_path.write_text(content, encoding='utf-8')
                print(f"      ✅ Created successfully!\n")
            else:
                print(f"      ⚠️  Failed to generate, skipping...\n")

    print("\n" + "="*60)
    print("✅ Base documentation generation complete!")
    print(f"📁 Location: {docs_dir.absolute()}")
    print(f"📊 Total lessons: {total_lessons}")
    print("\n🎯 Next steps:")
    print("   1. Review generated content")
    print("   2. Run generate_docs.py for personalization")
    print("   3. Run generate_urdu_docs.py for translation")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(generate_all_lessons())
