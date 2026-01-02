@(declarator)
declarator(decl, init) -> declarator(decl_1, init) 
   if decl.variable(vname) -> decl_1.variable(vname + "_1")