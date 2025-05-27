# # Cellular Sheaf DSL

# Here, we will illustrate how to create a cellular sheaf with the ```@cellular_sheaf``` macro.

# ## Introduction

# The Cellular Sheaf macro allows users to construct a cellular sheaf object. This implies that we can create a graph structure ```G(V,E)``` where V = Vertices and E = Edges in the graph "G". Then we can create a sheaf object on top of this graph where we can denote the following:

# - Vertex Stalks and their respective dimensions.
# - Edge Stalks and their respective dimensions.
# - Restriction Maps defined by matrices.

# ## Key Components

# The Cellular Sheaf macro allows a user to declare the following:

# - **Restriction Maps:** A user can define numerous restriction maps outside of the embedded DSL. Restriction maps are defined as Julia matrices:
# ```A = [1 0 0; 0 0 1]```
# - **Vertex Stalks:** A user can define numerous vertex stalks by declaring a variable a ```Stalk``` type in their variable declarations. To complete a ```Stalk``` type, the user must also specify the dimension of the given vector space. This can be done by specifying a digit within brackets after the type annotation: 
#    - For instance, with ```Stalk{1}```, ```1``` specifies the dimension of the vector space.
#    - To complete your vertex stalk, assign your typing to a variable name of choice such as ```vertex_name::Stalk{1}```.
# -  **Graph Edges:** Now that we have declared our restriction maps and our vertex stalks, what can we do with them? We can declare a system of linear equations that represents the relationships or edges between vertices.
#    - Let us say we are given restriction maps: A and B. Additionally, we are given vertex stalks x and y. We want to create an edge between x and y where the two incident vertices share an edge stalk vector space. We will need to map our given vertex stalks into the edge stalk vector space through our restriction maps: A and B. Given that A maps x to the shared edge space and B maps y to the shared edge space, we yield the following relation: ```A(x) == B(y)```.
#    - This declares an edge with incident vertex stalks x and y where A maps x to the shared edge space and B maps y to the shared edge space.

# ## Putting It All Together

# Now that we understand the different components of building a cellular sheaf within our DSL, let's create our very own cellular sheaf.

# First, let's import necessary modules:

using AlgebraicOptimization

# ### Triangular Sheaf

# Given the following graph: ```G(V, E)``` where:

# - ```V = {x, y, z}```
# - ```E = {[x -> y], [x -> z], [y -> z]}```

# We aim to create a cellular sheaf where x, y, and z are vertex stalks of dimension 4.

# Furthermore, we want to establish each edge stalk with a dimension of 1. To map our vertex stalk to our edge stalk, we will need a 1 x 4 matrix for each. We can define these as A, B, C with arbitrary values since we are more concerned with the relationships than the information in this guide. The information such as feature vectors and mappings are up to the end user.

# To create the restriction maps, we create Julia matrices as we normally would:

# ```
# A = [1 0 0 0]
# B = [1 0 0 0]
# C = [1 0 0 0]
# ```

# We can then pass these values into our macro as arguments as follows:

# ```
# sheaf = @cellular_sheaf A, B, C begin end
# ```

# This will essentially declare these restriction maps within our cellular sheaf.

# Now, let's declare our vertex stalks and their respective dimensions. We stated that we want to declare vertex stalks of dimension 4. We can do so within our macro scope as follows:

# ```
# sheaf = @cellular_sheaf A, B, C begin
#   x::Stalk{4}, y::Stalk{4}, z::Stalk{4}
# end
# ```

# Great, now we want to declare our edges in the graph as well as our relationship between our restriction maps and vertex stalks. We can do this using the relationship structure illustrated earlier:

# ```
# sheaf = @cellular_sheaf A, B, C begin
#     x::Stalk{4}, y::Stalk{4}, z::Stalk{4}
#     A(x) == B(y)
#     A(x) == C(z)
#     B(y) == C(z)
# end
# ```

# This establishes three edges within our graph.

# - An edge from x to y with restriction maps A and B.
# - An edge from x to z with restriction maps A and C.
# - An edge from y to z with restriction maps B and C.

# To conclude, a complete triangular cellular sheaf would look as follows:

A = [1 0 1 0]
B = [1 0 0 1]
C = [1 0 0 0]

macro_result = @cellular_sheaf A, B, C begin
    x::Stalk{4}, y::Stalk{4}, z::Stalk{4}

    A(x) == B(y)
    A(x) == C(z)
    B(y) == C(z)

end

# This translates to these lower-level functions:


sheaf = CellularSheaf([4, 4, 4], [1, 1, 1])
set_edge_maps!(sheaf, 1, 2, 1, A, B)
set_edge_maps!(sheaf, 1, 3, 2, A, C)
set_edge_maps!(sheaf, 2, 3, 3, B, C)


# which generates this sheaf object:

sheaf

# **Inferred Edge Stalks:** You may be wondering what happened to declaring edge stalks? Because restriction maps and vertex stalks are enough to infer the edge stalks dimensions, the end user does not need to worry about declaring edge stalks. They can focus on solely restriction maps and vertex stalks, minimizing potential errors form wrongly defined edge stalk dimensions.

# ## Potential Exceptions 

# - "No restriction maps were passed into the macro."
#    - The macro expects at least one restriction map input. This error implies you are trying to create a cellular sheaf with zero restriction maps.
# - "Restriction map ... is not a matrix"
#    - The macro expects the inputted restriction map to be a matrix. This error will throw when you attempt to pass in a restriction map that is not a matrix.
# - "Variable declaration: ... format is invalid."
#   - Indicates that the inputted declaration did not match expected declaration format.
# - "Term ... is an invalid product. A product is of form A*x or A(x)."
#   -  Indicates the LHS or RHS of the equation was not a valid product or function of restriction map and vertex stalk.
# - "Line is malformed"
#   - This is a more general parsing error stating that your inputted line of code did not parse as a declaration or an equation.
# - "Variable type ... is unsupported. Current types include: "Stalk" (Vertex Stalk)."
#   - Indicates you attempted to declare a variable with an undefined type.
# - "Variable: ... has already been declared."
#   - Indicates you are attempting to redeclare a variable. This is not allowed.
# - "Inferred edge stalk on relation: Ax = By" is inconsistent. Left restriction map maps dimension ... to dimension ... . Right restriction map maps dimension ... to dimension ... ."
#   - Indicates that the there is no possible edge stalk value to be inferred based on the restriction map and vertex values you declared.
# - "Left/Right restriction map (Size: ...) cannot map left/right vertex stalk (Dimension: ...)."
#   - Implies the restriction map vector space and vertex stalk vector space can not be multiplied.
# - "Restriction map "..." in ", Ax = By" is undefined."
#   - The restriction map used in the given equation was never declared.