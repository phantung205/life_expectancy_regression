from flask import Blueprint,render_template
from services import report_service

report_bp = Blueprint("report",__name__)

@report_bp.route("/reports")
def reports():
    result = report_service.get_report_names()
    return render_template("reports.html",reports=result)

