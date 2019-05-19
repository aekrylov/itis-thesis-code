from recommenders.webapp import app, db


class Document(db.Model):
    id = db.Column(db.Integer(), autoincrement=True, primary_key=True)
    processed_text = db.Column(db.UnicodeText(), nullable=False)

    # KAD data
    kad_case_num = db.Column(db.String(32), unique=True)
    kad_case_id = db.Column(db.String(36), unique=True)
    kad_doc_id = db.Column(db.String(36), unique=True)
    kad_doc_name = db.Column(db.String(128))


class Rating(db.Model):
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)

    doc_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    doc = db.relationship('Document', backref=db.backref('ratings'), foreign_keys=doc_id)

    recommendation_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    recommendation = db.relationship('Document', foreign_keys=recommendation_id)

    value = db.Column(db.SmallInteger, nullable=False)
    ip = db.Column(db.String(length=15), nullable=False)

    __table_args__ = (db.UniqueConstraint('doc_id', 'recommendation_id'), )


@app.cli.command()
def fill_db(metadata=None, data_samples=None):
    if metadata is None:
        from .webapp import metadata
    if data_samples is None:
        from .webapp import data_samples

    for doc_id in range(len(data_samples)):
        doc = Document(
            id=doc_id,
            processed_text=data_samples[doc_id],
            kad_case_num=metadata[doc_id]['case_num'],
            kad_case_id=metadata[doc_id]['case_id'],
            kad_doc_id=metadata[doc_id]['doc_id'],
            kad_doc_name=metadata[doc_id]['doc_name']
        )
        db.session.add(doc)

    db.session.commit()