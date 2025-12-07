import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_csv_report(prediction_data):
    """Generate CSV report from prediction data."""
    try:
        df = pd.DataFrame([prediction_data])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error generating CSV report: {e}")
        return None

def generate_pdf_report(prediction_data, username="User"):
    """Generate PDF report with prediction results."""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#FF4B4B'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story.append(Paragraph("🔋 Battery RUL Prediction Report", title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Metadata
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>User:</b> {username}", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))
        
        # Prediction Results
        story.append(Paragraph("<b>Prediction Results</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        # Create results table
        results_data = [
            ['Metric', 'Value'],
            ['Predicted RUL', f"{prediction_data.get('rul', 'N/A')} cycles"],
            ['Estimated Time', f"{prediction_data.get('time_estimate', 'N/A')} years"],
            ['Current Cycle', f"{prediction_data.get('cycle', 'N/A')}"],
            ['Temperature', f"{prediction_data.get('temperature', 'N/A')} °C"],
            ['Voltage', f"{prediction_data.get('voltage', 'N/A')} V"],
            ['Current', f"{prediction_data.get('current', 'N/A')} A"],
            ['Model Used', prediction_data.get('model', 'XGBoost')]
        ]
        
        results_table = Table(results_data, colWidths=[3*inch, 3*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF4B4B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(results_table)
        story.append(Spacer(1, 0.5 * inch))
        
        # Recommendations
        story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        recommendations = [
            "Monitor battery health regularly to track degradation trends.",
            "Avoid extreme temperatures to extend battery life.",
            "Maintain optimal charge rates (0.5C - 1C) for longevity.",
            "Plan replacement before reaching 80% capacity threshold.",
            "Consider recycling options to reduce environmental impact."
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None