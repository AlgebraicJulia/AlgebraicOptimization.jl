module TestADT

using Test
using AlgebraicOptimization
using AlgebraicOptimization.HomologicalProgramming.CellularSheafTerm:
    Declaration, RestrictionMap, VertexStalk, TypeName, Product, Equation, UntypedDeclaration, TypedDeclaration, CellularSheafExpr, construct

# Let's prove that the current AST properly represents a cellular sheaf

### Judgments:

# Restriction Maps
A = UntypedDeclaration(:A, [1 0 0 0])
B = UntypedDeclaration(:B, [1 0 0 0])
C = UntypedDeclaration(:C, [1 0 0 0])

# Stalks
generic_type = TypeName(:Stalk, 4)

x = TypedDeclaration(:x, generic_type, nothing)
y = TypedDeclaration(:y, generic_type, nothing)
z = TypedDeclaration(:z, generic_type, nothing)

### Products

# Restriction Maps 
A_rm = RestrictionMap(:A, [1 0 0 0])
B_rm = RestrictionMap(:B, [1 0 0 0])
C_rm = RestrictionMap(:C, [1 0 0 0])


# Vertex Stalks
x_stalk = VertexStalk(:x, 4)
y_stalk = VertexStalk(:y, 4)
z_stalk = VertexStalk(:z, 4)

Ax = Product(A_rm, x_stalk)
By = Product(B_rm, y_stalk)
Cz = Product(C_rm, z_stalk)

### Equations

EQ1 = Equation(Ax, By)
EQ2 = Equation(By, Cz)
EQ3 = Equation(Cz, Ax)

### CellularSheafExpr

triangularSheaf = CellularSheafExpr([A, B, C, x, y, z], [EQ1, EQ2, EQ3])

# Testing no duplicate variables!
triangularSheafDuplicate = CellularSheafExpr([A, A, B, C, x, y, z], [EQ1, EQ2, EQ3])
@test_throws SheafDeclarationError("Variable: \"A\" has already been declared.") construct(triangularSheafDuplicate)

# Testing undeclared variable in equation
R_rm = RestrictionMap(:R, [1 0 0 0])
x_stalk = VertexStalk(:x, 4)

Rx = Product(R_rm, x_stalk)
EQ_undefined = Equation(Rx, By)

triangularSheafUndeclared = CellularSheafExpr([A, B, C, x, y, z], [EQ_undefined, EQ2, EQ3])
@test_throws SheafDeclarationError("Restriction map \"R\" in \"Rx = By\" is undefined.") construct(triangularSheafUndeclared)

# Test inconsistent edge stalk inferred from bad restriction maps
B_inconsistent = UntypedDeclaration(:B, [1 0 0 0; 0 0 0 1])

triangularSheafInconsistent = CellularSheafExpr([A, B_inconsistent, C, x, y, z], [EQ1, EQ2, EQ3])
@test_throws SheafDimensionMismatchError(
    """Inferred edge stalk on relation: "Ax = By" is inconsistent.
        Left restriction map maps dimension 4 to dimension 1.
        Right restriction map maps dimension 4 to dimension 2.
    """) construct(triangularSheafInconsistent)

# Incorrect vertex stalk dimension or restriction map size
B_bad_map = UntypedDeclaration(:B, [1 0 0])

triangularSheafWrongMapping = CellularSheafExpr([A, B_bad_map, C, x, y, z], [EQ1, EQ2, EQ3])
@test_throws SheafDimensionMismatchError("Right restriction map (Size: (1, 3)) cannot map right vertex stalk (Dimension: 4).") construct(triangularSheafWrongMapping)

end