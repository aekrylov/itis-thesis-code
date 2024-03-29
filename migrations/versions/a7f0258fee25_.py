"""empty message

Revision ID: a7f0258fee25
Revises: b3a60d8f67ce
Create Date: 2019-05-19 13:46:16.424714

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a7f0258fee25'
down_revision = 'b3a60d8f67ce'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint(None, 'rating', ['doc_id', 'recommendation_id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'rating', type_='unique')
    # ### end Alembic commands ###
