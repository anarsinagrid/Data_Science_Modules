-- COMBINE TWO TABLES
SELECT 
    firstname,
    lastname,
    city,
    state 
FROM Person 
LEFT JOIN Address 
    ON Person.personId = Address.personId;

-- SECOND HIGHEST SALARY
SELECT 
    (SELECT DISTINCT salary
     FROM Employee
     ORDER BY salary DESC
     LIMIT 1 OFFSET 1) AS SecondHighestSalary;

-- NTH HIGHEST SALARY
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    SET N = N - 1;
    RETURN (
        SELECT DISTINCT salary 
        FROM Employee 
        ORDER BY salary DESC 
        LIMIT 1 OFFSET N
    );
END

-- RANK SCORES
SELECT 
    s1.score,
    (SELECT COUNT(DISTINCT s2.score)
     FROM Scores s2
     WHERE s2.score >= s1.score) AS 'rank'
FROM Scores s1
ORDER BY s1.score DESC;

-- CONSECUTIVE NUMBERS
SELECT DISTINCT l1.num AS ConsecutiveNums
FROM Logs l1, Logs l2, Logs l3
WHERE l1.id = l2.id - 1
  AND l2.id = l3.id - 1
  AND l1.num = l2.num
  AND l2.num = l3.num;

-- EMPLOYEES EARNING MORE THAN THEIR MANAGERS
SELECT a.name AS Employee
FROM Employee a
JOIN Employee b 
    ON a.managerId = b.id
WHERE a.salary > b.salary;

-- DUPLICATE EMAILS
SELECT DISTINCT a.email AS Email
FROM Person a
JOIN Person b 
    ON a.email = b.email 
    AND a.id <> b.id;

-- CUSTOMERS WHO NEVER ORDERED
SELECT c.name AS Customers
FROM Customers c
LEFT JOIN Orders o
    ON c.id = o.customerId
WHERE o.customerId IS NULL;

-- DEPARTMENT HIGHEST SALARY
SELECT 
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM Employee e
INNER JOIN Department d 
    ON e.departmentId = d.id
WHERE e.salary = (
    SELECT MAX(salary) 
    FROM Employee 
    WHERE departmentId = d.id
);

-- EXCHANGE SEATS
SELECT
    CASE 
        WHEN id = (SELECT MAX(id) FROM Seat) AND id % 2 = 1 THEN id
        WHEN id % 2 = 1 THEN id + 1 
        ELSE id - 1 
    END AS id,
    student
FROM Seat
ORDER BY id;
