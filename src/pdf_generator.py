import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(applicant_name, assessment_id, risk_info, applicant_dict, officer_name, officer_notes="") -> bytes:
    """
    Generates an official Credit Underwriting Assessment Dossier in PDF format.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a')
    )
    
    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b')
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    story = []
    
    # Header
    story.append(Paragraph("INSTITUTIONAL CREDIT RISK ASSESSMENT MEMORANDUM", title_style))
    story.append(Paragraph(f"AI-Driven Underwriting Intelligence System • Document ID: {assessment_id}", subtitle_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0f172a'), spaceAfter=14))
    
    # Status Banner
    score = risk_info.get('score', 50.0)
    tier = risk_info.get('tier', 'Medium Risk')
    action = risk_info.get('action', 'Review')
    
    if score <= 30.0:
        stamp_color = colors.HexColor('#10b981')
        verdict = "APPROVED (RECOMMENDED)"
    elif score <= 70.0:
        stamp_color = colors.HexColor('#f59e0b')
        verdict = "UNDER REVIEW (ADDITIONAL DOCS REQUIRED)"
    else:
        stamp_color = colors.HexColor('#ef4444')
        verdict = "REJECTED (HIGH DEFAULT PROBABILITY)"
        
    status_data = [
        [
            Paragraph(f"<b>RISK SCORE:</b> <font size=14 color='{stamp_color.hexval()}'><b>{score:.1f} / 100</b></font>", body_style),
            Paragraph(f"<b>RISK TIER:</b> <b>{tier}</b>", body_style),
            Paragraph(f"<b>DECISION VERDICT:</b> <font color='{stamp_color.hexval()}'><b>{verdict}</b></font>", body_style)
        ]
    ]
    t_status = Table(status_data, colWidths=[2.2*inch, 2.0*inch, 3.0*inch])
    t_status.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, stamp_color),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_status)
    story.append(Spacer(1, 14))
    
    # Applicant Profile Table
    profile_data = [
        [Paragraph("<b>Applicant Details</b>", body_style), "", Paragraph("<b>Credit Facility Terms</b>", body_style), ""],
        [Paragraph("<b>Name:</b>", body_style), Paragraph(str(applicant_name), body_style),
         Paragraph("<b>Requested Amount:</b>", body_style), Paragraph(f"${applicant_dict.get('loan_amnt', 0):,.2f}", body_style)],
        [Paragraph("<b>Age / Tenure:</b>", body_style), Paragraph(f"{applicant_dict.get('person_age')} yrs / {applicant_dict.get('person_emp_length')} yrs", body_style),
         Paragraph("<b>Interest Rate / Grade:</b>", body_style), Paragraph(f"{applicant_dict.get('loan_int_rate')}% (Grade {applicant_dict.get('loan_grade')})", body_style)],
        [Paragraph("<b>Annual Income:</b>", body_style), Paragraph(f"${applicant_dict.get('person_income', 0):,.2f}", body_style),
         Paragraph("<b>Loan Purpose:</b>", body_style), Paragraph(str(applicant_dict.get('loan_intent')), body_style)],
        [Paragraph("<b>Housing Status:</b>", body_style), Paragraph(str(applicant_dict.get('person_home_ownership')), body_style),
         Paragraph("<b>Debt-to-Income (DTI):</b>", body_style), Paragraph(f"{applicant_dict.get('loan_percent_income', 0):.1%}", body_style)],
        [Paragraph("<b>Bureau Defaults:</b>", body_style), Paragraph("Yes (Record Found)" if applicant_dict.get('cb_person_default_on_file') == 'Y' else "No (Clean)", body_style),
         Paragraph("<b>Credit History:</b>", body_style), Paragraph(f"{applicant_dict.get('cb_person_cred_hist_length')} years", body_style)]
    ]
    
    t_profile = Table(profile_data, colWidths=[1.4*inch, 2.2*inch, 1.6*inch, 2.0*inch])
    t_profile.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (2,0), (3,0), colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('SPAN', (0,0), (1,0)),
        ('SPAN', (2,0), (3,0)),
    ]))
    story.append(t_profile)
    story.append(Spacer(1, 14))
    
    # Underwriting Analysis Notes
    story.append(Paragraph("<b>Underwriting Audit & Explainable AI Rationale:</b>", body_style))
    analysis_text = f"""
    The credit risk prediction model utilizes gradient boosted decision trees (XGBoost) evaluated against 32,500+ records.
    The applicant's default probability is estimated at {risk_info.get('probability', 0):.2%}. Key predictive drivers include 
    Loan Grade ({applicant_dict.get('loan_grade')}), Debt-to-Income ratio ({applicant_dict.get('loan_percent_income', 0):.1%}), 
    and historical default track record. Officer Notes: {officer_notes or 'Standard assessment protocol followed.'}
    """
    story.append(Paragraph(analysis_text, body_style))
    story.append(Spacer(1, 24))
    
    # Signature Lines
    sig_data = [
        [Paragraph("___________________________________<br/><b>Evaluating Credit Officer</b><br/>" + str(officer_name), body_style),
         Paragraph("___________________________________<br/><b>Authorized Credit Committee Chair</b><br/>Credit Approval Authority", body_style)]
    ]
    t_sig = Table(sig_data, colWidths=[3.6*inch, 3.6*inch])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_sig)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
